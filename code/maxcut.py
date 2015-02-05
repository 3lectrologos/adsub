import os
import numpy as np
import igraph as ig
import submod
import util


def cutval(g, cut):
    cut = set(cut)
    rest = set(g.vs.indices) - cut
    return len(g.es.select(_between=(cut, rest)))


def cutdif(g, u, cut, cut_edges):
    if u in cut:
        return (0, set(), set())
    cut_edges = set(cut_edges)
    u_edges = set(g.incident(u))
    added_edges = u_edges - cut_edges
    removed_edges = cut_edges & u_edges
    return (len(added_edges) - len(removed_edges), added_edges, removed_edges)


def pcut(g, cset, cut, cut_edges):
    cut_edges = set(cut_edges)
    dif = []
    for u in cset:
        (diflen, _, _) = cutdif(g, u, cut, cut_edges)
        dif.append(diflen)
    return np.mean(dif)


class NonAdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, nsim):
        super(NonAdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.ins = [(set(), set(), 0)] * nsim

    def mean_pcut(self, cid):
        return np.mean([pcut(self.g, self.csets[cid], cut, cut_edges) + val
                        for (cut, cut_edges, val) in self.ins])

    def init_f_hook(self):
        super(NonAdaptiveMaxCut, self).init_f_hook()
        self.f = lambda v, a: self.mean_pcut(v)

    def update_f_hook(self):
        #        print 'nonad:', len(self.sol)
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        cset = self.csets[self.sol[-1]]
        for i, (cut, cut_edges, val) in enumerate(self.ins):
            vcut = np.random.choice(cset)
            elen, eadd, erem = cutdif(self.g, vcut, cut, cut_edges)
            new_val = val + elen
            new_cut = cut | set([vcut])
            new_cut_edges = (cut_edges - erem) | eadd
            self.ins[i] = (new_cut, new_cut_edges, new_val)


class AdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, instance):
        super(AdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.instance = instance

    def init_f_hook(self):
        super(AdaptiveMaxCut, self).init_f_hook()
        self.cut = set()
        self.cut_edges = set()
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut, self.cut_edges)

    def update_f_hook(self):
        vcut = self.instance[self.sol[-1]]
        elen, eadd, erem = cutdif(self.g, vcut, self.cut, self.cut_edges)
        self.fsol += elen
        self.cut.add(vcut)
        self.cut_edges = (self.cut_edges - erem) | eadd
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut, self.cut_edges) + self.fsol


def random_instance(csets):
    inst = {}
    for cid in csets:
        inst[cid] = np.random.choice(csets[cid])
    return inst


def eval_instance(g, inst, cutids):
    cut = [inst[cid] for cid in cutids]
    return cutval(g, cut)


def compare(g, csets, k, nsim_nonad, niter):
    f_rand = []
    f_nonad = []
    f_ad = []
    solver_nonad = NonAdaptiveMaxCut(g, csets, nsim_nonad)
    cut_rand, _ = solver_nonad.random(k)
    solver_nonad_new = NonAdaptiveMaxCut(g, csets, nsim_nonad)
    cut_nonad, _ = solver_nonad_new.random_greedy(k)
    for i in range(niter):
        instance = random_instance(csets)
        solver_ad = AdaptiveMaxCut(g, csets, instance)
        cut_ad, _ = solver_ad.random_greedy(k)
        f_rand.append(eval_instance(g, instance, cut_rand))
        f_nonad.append(eval_instance(g, instance, cut_nonad))
        f_ad.append(eval_instance(g, instance, cut_ad))
    fm_rand = np.mean(f_rand)
    fm_nonad = np.mean(f_nonad)
    fm_ad = np.mean(f_ad)
    return {
        'f_rand': fm_rand,
        'f_nonad': fm_nonad,
        'f_ad': fm_ad
        }


def all_csets(g):
    return {str(v.index): [v.index] + g.neighbors(v) for v in g.vs}


def k_csets(g, k):
    vs = np.random.choice(g.vs, k, replace=False)
    return {str(v.index): [v.index] + g.neighbors(v) for v in vs}


def run(g, reps, n_available, k, niter, nsim_nonad):
    res = {'f_rand': [], 'f_nonad': [], 'f_ad': []}
    for rep in range(reps):
        print 'Rep:', rep, '(k = {0})'.format(k)
        print '==============================='
        r = compare(g, k_csets(g, n_available), k, nsim_nonad, niter)
        res['f_rand'].append(r['f_rand'])
        res['f_nonad'].append(r['f_nonad'])
        res['f_ad'].append(r['f_ad'])
        print 'f_rand =', r['f_rand']
        print 'f_nonad =', r['f_nonad']
        print 'f_ad =', r['f_ad']
        print '===============================\n'
    return res

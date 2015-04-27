import os
import sys
import random
import igraph as ig
import numpy as np
import joblib
import bitarray as ba
import progressbar
import submod
import util


DEBUG = False


def random_instance(g, p, copy=False, ret=False):
    if copy: g = g.copy()
    to_remove = []
    for e in g.es:
        if random.random() > p:
            to_remove.append(e.tuple)
    g.delete_edges(to_remove)
    if ret:
        return (g, to_remove)
    else:
        return g


def ic(g, a):
    r = list(set(a))
    for v in a:
        r.extend(g.subcomponent(v, mode=ig.OUT))
        # Just to avoid the list getting too long
        r = list(set(r))
    return set(r)


def ic_sim(g, p, niter, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    csim = {v['i']: {} for v in g.vs}
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in g.vs:
            b = n*ba.bitarray('0')
            for z in g.subcomponent(v, mode=ig.OUT):
                b[z] = True
            csim[v['i']][i] = b
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim


def ic_sim_cond(g, p, niter, active):
    n = g.vcount()
    csim = {v.index: 0 for v in g.vs}
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in csim:
            csim[v] += len(g.subcomponent(v, mode=ig.OUT))
        g.add_edges(rem)
    d = {}
    for v in g.vs:
        d[v['i']] = len(active) + csim[v.index]/(1.0*niter)
    return d


def delete_active(g, active):
    g.delete_vertices([v for v in g.vs if v['i'] in active])


def f_comb(val, cost, GAMMA):
    return val - GAMMA*cost


def f_ic_base(h, a, fc):
    active = ic(h, a)
    return fc(len(active), len(a))


def f_ic(v, a, csim, prev, fc, ret=False):
    s = 0
    vcount = len(csim)
    nsim = len(csim[v])
    if prev != None:
        for k, b in csim[v].iteritems():
            if ret:
                prev[k] |= b
                s += prev[k].count(1)
            else:
                s += (prev[k] | b).count(1)
    else:
        if ret:
            prev = {i: vcount*ba.bitarray('0') for i in range(nsim)}
        for k, b, in csim[v].iteritems():
            s += b.count(1)
            if ret:
                prev[k] |= b
    if ret:
        return (fc((1.0*s)/nsim, len(a) + 1), prev)
    else:
        return fc((1.0*s)/nsim, len(a) + 1)


def f_ic_ad(v, a, csim, active, fprev, fc):
    a = set(a)
    active = list(set(active))
    if v in active:
        return fprev
    return fc(csim[v], len(a) + 1)


class NonAdaptiveInfluence(submod.AdaptiveMax):
    def __init__(self, g, gset, p, fc, nsim):
        super(NonAdaptiveInfluence, self).__init__(gset)
        self.g = g
        self.p = p
        self.fc = fc
        self.nsim = nsim
        self.csim = ic_sim(self.g, self.p, self.nsim, pbar=True)

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.prev = None
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev, self.fc)

    def update_f_hook(self):
        (self.fsol, self.prev) = f_ic(self.sol[-1], self.sol[:-1], self.csim,
                                      self.prev, self.fc, ret=True)
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev, self.fc)


class AdaptiveInfluence(submod.AdaptiveMax):
    def __init__(self, g, gset, h, p, fc, nsim):
        super(AdaptiveInfluence, self).__init__(gset)
        self.g = g.copy()
        self.h = h
        self.p = p
        self.fc = fc
        self.nsim = nsim

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic(self.h, self.sol)
        delete_active(self.g, active)
        csim = ic_sim_cond(self.g, self.p, self.nsim, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol, self.fc)
        self.fsol = self.fc(len(active), len(self.sol))


def worker(i, g, gset, pedge, k, gamma, nsim_ad, v_rand, v_nonad):
    sys.stdout.write('.')
    sys.stdout.flush()
    fc = lambda a, b: f_comb(a, b, gamma)
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, gset, h, pedge, fc, nsim_ad)
    (v_ad, _) = solver_ad.random_greedy(k)
    if f_ic_base(h, v_ad, fc) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    solver_ad = AdaptiveInfluence(g, gset, h, pedge, fc, nsim_ad)
    (v_ad_g, _) = solver_ad.greedy(k)
    return {
        'f_rand': f_ic_base(h, v_rand, fc),
        'f_nonad': f_ic_base(h, v_nonad, fc),
        'f_ad': f_ic_base(h, v_ad, fc),
        'f_ad_g': f_ic_base(h, v_ad_g, fc)
        }


def compare(g, pedge, n_available, k ,gamma, nsim_nonad, nsim_ad, niter, workers):
        gset = [v['i'] for v in np.random.choice(g.vs, n_available, replace=False)]
        # Non-adaptive simulation
        fc = lambda a, b: f_comb(a, b, gamma)
        solver_nonad = NonAdaptiveInfluence(g, gset, pedge, fc,  nsim_nonad)
        (v_rand, _) = solver_nonad.random(k)
        (v_nonad, _) = solver_nonad.random_greedy(k)
        del solver_nonad.csim
        # Adaptive simulation
        arg = [g, gset, pedge, k, gamma, nsim_ad, v_rand, v_nonad]
        if workers > 1:
            r = joblib.Parallel(n_jobs=workers)((worker, [i] + arg, {})
                                                  for i in range(niter))
        else:
            r = [worker(*([i] + arg)) for i in range(niter)]
        return {
            'f_rand': np.mean([x['f_rand'] for x in r]),
            'f_nonad': np.mean([x['f_nonad'] for x in r]),
            'f_ad': np.mean([x['f_ad'] for x in r]),
            'f_ad_g': np.mean([x['f_ad_g'] for x in r])
            }


def run(g, reps, pedge, n_available, k, gamma, nsim_nonad, nsim_ad, niter, workers=0):
    res = {'f_rand': [], 'f_nonad': [], 'f_ad': [], 'f_ad_g': []}
    for rep in range(reps):
        print 'Rep:', rep, '(k = {0}, p = {1})'.format(k, pedge)
        print '==============================='
        r = compare(g, pedge, n_available, k, gamma, nsim_nonad, nsim_ad, niter, workers)
        res['f_rand'].append(r['f_rand'])
        res['f_nonad'].append(r['f_nonad'])
        res['f_ad'].append(r['f_ad'])
        res['f_ad_g'].append(r['f_ad_g'])
        print ''
        print 'f_rand =', r['f_rand']
        print 'f_nonad =', r['f_nonad']
        print 'f_ad =', r['f_ad']
        print 'f_ad_g =', r['f_ad_g']
        print '===============================\n'
    return res

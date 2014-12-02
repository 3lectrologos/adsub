import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import submod


def cutval(g, cut):
    cut = set(cut)
    val = 0
    for _, w2 in g.edges_iter(cut):
        if w2 not in cut:
            val += 1
    return val

def cutdif(g, u, cut):
    cut = set(cut)
    if u in cut:
        return 0
    val = 0
    for _, w2 in g.edges_iter(u):
        if w2 in cut:
            val -= 1
        else:
            val += 1
    return val

def pcut(g, cs, cut):
    cut = set(cut)
    dif = []
    for u in cs:
        dif.append(cutdif(g, u, cut))
    return np.mean(dif)

class NonAdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, nsim):
        super(NonAdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.ins = [(set(), 0)] * nsim

    def mean_pcut(self, cid):
        return np.mean([pcut(self.g, self.csets[cid], cut) + val
                        for (cut, val) in self.ins])

    def init_f_hook(self):
        super(NonAdaptiveMaxCut, self).init_f_hook()
        self.f = lambda v, a: self.mean_pcut(v)

    def update_f_hook(self):
        print 'nonad:', len(self.sol)
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        cs = self.csets[self.sol[-1]]
        for i, (cut, val) in enumerate(self.ins):
            vcut = np.random.choice(cs)
            newcut = cut | set([vcut])
            newval = cutdif(self.g, vcut, cut) + val
            self.ins[i] = (newcut, newval)

class AdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, instance):
        super(AdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.instance = instance

    def init_f_hook(self):
        super(AdaptiveMaxCut, self).init_f_hook()
        self.cut = set()
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut)

    def update_f_hook(self):
        vcut = self.instance[self.sol[-1]]
        self.fsol += cutdif(self.g, vcut, self.cut)
        self.cut.add(vcut)
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut) + self.fsol

def random_instance(csets):
    inst = {}
    for cid in csets:
        inst[cid] = np.random.choice(csets[cid])
    return inst

def eval_instance(g, inst, cutids):
    cut = [inst[cid] for cid in cutids]
    return cutval(g, cut)

def compare(g, csets, nsim_nonad, niter):
    f_nonad = []
    f_ad = []
    solver_nonad = NonAdaptiveMaxCut(g, csets, nsim_nonad)
    cut_nonad, _ = solver_nonad.random_greedy(len(csets))
    for i in range(niter):
        print 'i =', i
        instance = random_instance(csets)
        solver_ad = AdaptiveMaxCut(g, csets, instance)
        cut_ad, _ = solver_ad.random_greedy(len(csets))
        f_nonad.append(eval_instance(g, instance, cut_nonad))
        f_ad.append(eval_instance(g, instance, cut_ad))
    print 'f_nonad =', np.mean(f_nonad)
    print 'f_ad =', np.mean(f_ad)

def test_graph():
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(1, 5)
    g.add_edge(1, 6)
    return g

def std_csets(g):
    return {str(v): [v] + list(nx.all_neighbors(g, v)) for v in g.nodes_iter()}

if __name__ == "__main__":
    NSIM_NONAD = 100
    NITER = 100
#    g = test_graph()
    g = nx.barabasi_albert_graph(100, 2)
    compare(g, std_csets(g), NSIM_NONAD, NITER)

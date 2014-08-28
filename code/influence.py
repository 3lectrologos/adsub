import os
import sys
import random
import cProfile as prof

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import bitarray as ba
import progressbar

import submod
import util


def random_instance(g, p, copy=False):
    if copy: g = g.copy()
    for e in g.edges():
        if random.random() > p:
            g.remove_edge(*e)
    return g

def random_ind(g, p):
    h = {}
    gedges = g.edges()
    for v in g.nodes_iter():
        random_instance(g, p)
        h[v] = set(ic_one(g, v))
        g.add_edges_from(gedges)
    return h

def ic_one(g, v):
    return nx.shortest_path_length(g, v).keys()

def ic_sim(g, p, niter, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.number_of_nodes()
    csim = np.zeros(shape=(n, n, niter), dtype='bool')
    gedges = g.edges()
    rp = {}
    for v in g.nodes_iter():
        rp[v] = np.random.permutation(niter)
    for i in range(niter):
        random_instance(g, p, copy=False)
        sp = nx.all_pairs_shortest_path_length(g)
        for v in g.nodes_iter():
            # Put in random position
            csim[v, sp[v].keys(), rp[v][i]] = True
        g.add_edges_from(gedges)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_base(h, a):
    active = set()
    for v in a:
        active |= h[v]
    return active

# Global weighting factor of node cost
LAMBDA = 3.0

def f_ic_base(h, a):
    active = ic_base(h, a)
    return len(active) - LAMBDA*len(a)

def f_ic(a, csim):
    a = list(set(a))
    if a == []: return 0
    act = csim[a[0], :, :]
    for v in a[1:]:
        act = act | csim[v, :, :]
    return (1.0*np.sum(act))/csim.shape[2] - LAMBDA*len(a)

def f_ic_ad(v, a, csim, active):
    a = set(a)
    active = list(set(active))
    # Elements in `active` repeated for each iteration of csim
    act = np.zeros(shape=(csim.shape[1], 1), dtype='bool')
    act[active] = True
    act = np.tile(act, (1, csim.shape[2]))
    # Elements in `active` union with anything that can be reached by `v`
    act = act | csim[v, :, :]
    return (1.0*np.sum(act))/csim.shape[2] - LAMBDA*(len(a) + 1)

# vals is a dictionary from nodes (not necessarily all of them) to "strengths"
def draw_alpha(g, vals, pos=None, maxval=None):
    if not maxval: maxval = max(a.values())
    if not pos: pos = nx.spring_layout(g)
    nx.draw_networkx_edges(g, pos,
                           edge_color='#cccccc',
                           alpha=0.5,
                           arrows=False)
    for v in g.nodes_iter():
        if v not in vals or vals[v] == 0:
            nx.draw_networkx_nodes(g, pos, nodelist=[v],
                                   node_color='#555555',
                                   alpha=0.5)
        else:
            nx.draw_networkx_nodes(g, pos, nodelist=[v],
                                   node_color='b',
                                   alpha=(1.0*vals[v])/maxval)
    nx.draw_networkx_labels(g, pos, nodelist=[v],
                            font_size=10,
                            font_color='#eeeeee')

class BaseInfluence(submod.AdaptiveMax):
    pass

class NonAdaptiveInfluence(BaseInfluence):
    def __init__(self, csim):
        super(NonAdaptiveInfluence, self).__init__(g.nodes())
        self.csim = csim

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.f = lambda v, a: f_ic(a + [v], self.csim)

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, csim):
        super(AdaptiveInfluence, self).__init__(g.nodes())
        self.csim = csim

    def random_greedy(self, h, k):
        self.h = h
        return super(AdaptiveInfluence, self).random_greedy(k)

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic_base(self.h, self.sol)
        self.f = lambda v, a: f_ic_ad(v, a, self.csim, active)
        self.fsol = len(active) - LAMBDA*len(self.sol)

def test_graph():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(4, 1)
    g.add_edge(4, 2)
    g.add_edge(4, 3)
    g.add_edge(5, 6)
    return g.to_directed()

# Ratio of random greedy cardinality constraint
K_RATIO = 1

def compare_worker(i, g, pedge, vrg_nonad, solver_ad):    
    print '-> worker', i, 'started.'
    h = random_ind(g, pedge)
    (vrg_ad, _) = solver_ad.random_greedy(h, g.number_of_nodes()/K_RATIO)
    active_nonad = ic_base(h, vrg_nonad)
    active_ad = ic_base(h, vrg_ad)
    eval1 = f_ic_base(h, vrg_ad)
    eval2 = solver_ad.fsol
    print 'vs =', vrg_ad
    print 'val =', eval1, ',', eval2
    if eval1 != eval2: raise 'Inconsistent adaptive function values'
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': len(vrg_ad),
            'f_nonad': f_ic_base(h, vrg_nonad),
            'f_ad': f_ic_base(h, vrg_ad)}

def compare(g, pedge, nsim, niter, parallel=True, plot=False, savefig=False):
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.nodes_iter():
        st_nonad[v] = 0
        st_ad[v] = 0
    # Non-adaptive simulation
    csim = ic_sim(g, pedge, nsim, pbar=True)
    solver_nonad = NonAdaptiveInfluence(csim)
    (vrg_nonad, _) = solver_nonad.random_greedy(g.number_of_nodes()/K_RATIO)
    solver_ad = AdaptiveInfluence(csim)
    # Adaptive simulation
    arg = [g, pedge, vrg_nonad, solver_ad]
    if parallel:
        res = joblib.Parallel(n_jobs=4)((compare_worker, [i] + arg, {})
                                        for i in range(niter))
    else:
        res = [compare_worker(*([i] + arg)) for i in range(niter)]
    # Adjust strengths of active nodes
    for r in res:
        for v in r['active_nonad']:
            st_nonad[v] += 1
        for v in r['active_ad']:
            st_ad[v] += 1
    # Print results
    print 'Non-adaptive | favg =', np.mean([r['f_nonad'] for r in res]),
    print ',     #nodes =', len(vrg_nonad)
    print 'Adaptive     | favg =', np.mean([r['f_ad'] for r in res]),
    print ', avg #nodes =', np.mean([r['v_ad'] for r in res])
    pos = nx.spring_layout(g)
    # Plotting
    if plot:
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.gcf().set_size_inches(16, 9)
        plt.sca(ax1)
        ax1.set_aspect('equal')
        draw_alpha(g, st_nonad, pos=pos, maxval=niter)
        plt.sca(ax2)
        ax2.set_aspect('equal')
        draw_alpha(g, st_ad, pos=pos, maxval=niter)
        figname = 'INF'
        figname += '_P_EDGE_' + str(pedge*100)
        figname += '_NSIM_' + str(nsim)
        figname += '_NITER_' + str(niter)
        if SAVEFIG:
            plt.savefig(os.path.abspath('../results/' + figname + '.pdf'),
                        orientation='landscape',
                        papertype='letter',
                        bbox_inches='tight',
                        format='pdf')
        plt.show()

def profile_aux():
    g = nx.barabasi_albert_graph(50, 2)
    P_EDGE = 0.4
    NSIM = 1000
    NITER = 10
    PARALLEL = False
    PLOT = False
    SAVEFIG = False
    compare(g, P_EDGE, NSIM, NITER, PARALLEL, PLOT, SAVEFIG)

def profile():
    prof.run('influence.profile_aux()', sort='time')

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    g = nx.barabasi_albert_graph(500, 2)
    P_EDGE = 0.3
    NSIM = 1000
    NITER = 50
    PARALLEL = True
    PLOT = False
    SAVEFIG = False
    compare(g, P_EDGE, NSIM, NITER, PARALLEL, PLOT, SAVEFIG)

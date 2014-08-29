import argparse
import os
import sys
import random
import cProfile as prof

import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import joblib
import bitarray as ba
import progressbar

import submod
import util

DEBUG = False


def random_instance(g, p, active=None, copy=False, ret=False):
    if copy: g = g.copy()
    to_remove = []
    for e in g.es:
        if active == None or e.source not in active:
            if random.random() > p:
                to_remove.append(e.tuple)
    g.delete_edges(to_remove)
    if ret:
        return (g, to_remove)
    else:
        return g

def ic(g, a):
    a = list(set(a))
    if a == []: return set()
    b = np.array(g.shortest_paths()) != float('inf')
    idx = np.any(b[a, :], 0)
    return set(np.arange(g.vcount())[idx])

def ic_sim(g, p, niter, active=None, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    csim = np.zeros(shape=(n, n, niter), dtype='bool')
    for i in range(niter):
        (g, rem) = random_instance(g, p, active=active, copy=False, ret=True)
        sp = g.shortest_paths()
        csim[:, :, i] = (np.array(sp) != float('inf'))
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_sim_cond(g, p, niter, active, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    rest = [v.index for v in g.vs if v.index not in active]
    csim = {v: 0 for v in rest}
    bact = np.zeros(shape=g.vcount(), dtype='bool')
    bact[list(active)] = True
    for i in range(niter):
        (g, rem) = random_instance(g, p, active=active, copy=False, ret=True)
        for v in rest:
            sp = np.array(g.shortest_paths(source=v)[0]) != float('inf')
            csim[v] += np.sum(sp | bact)
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    for v in csim:
        csim[v] /= (1.0*niter)
    if pbar:
        pb.finish()
    return csim

# Global weighting factor of node cost
LAMBDA = 1.0

def f_ic_base(h, a):
    active = ic(h, a)
    return len(active) - LAMBDA*len(a)

def f_ic(v, a, csim, prev):
    if prev == None:
        act = csim[v, :, :]
    else:
        act = prev | csim[v, :, :]
    return (1.0*np.sum(act))/csim.shape[2] - LAMBDA*(len(a) + 1)

def f_ic_ad(v, a, csim, active, fprev):
    a = set(a)
    active = list(set(active))
    if v in active:
        return fprev
    return csim[v] - LAMBDA*(len(a) + 1)

# vals is a dictionary from nodes (not necessarily all of them) to "strengths"
# def draw_alpha(g, vals, pos=None, maxval=None):
#     if not maxval: maxval = max(a.values())
#     if not pos: pos = nx.spring_layout(g)
#     nx.draw_networkx_edges(g, pos,
#                            edge_color='#cccccc',
#                            alpha=0.5,
#                            arrows=False)
#     for v in g.nodes():
#         if v not in vals or vals[v] == 0:
#             nx.draw_networkx_nodes(g, pos, nodelist=[v],
#                                    node_color='#555555',
#                                    alpha=0.5)
#         else:
#             nx.draw_networkx_nodes(g, pos, nodelist=[v],
#                                    node_color='b',
#                                    alpha=(1.0*vals[v])/maxval)
#     nx.draw_networkx_labels(g, pos, nodelist=[v],
#                             font_size=10,
#                             font_color='#eeeeee')

class BaseInfluence(submod.AdaptiveMax):
    pass

class NonAdaptiveInfluence(BaseInfluence):
    def __init__(self, g, p, nsim):
        super(NonAdaptiveInfluence, self).__init__([v.index for v in g.vs])
        self.g = g
        self.p = p
        self.nsim = nsim

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.csim = ic_sim(self.g, self.p, self.nsim, pbar=True)
        self.prev = None
        self.fsol = 0
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev)

    def update_f_hook(self):
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        if self.prev == None:
            self.prev = self.csim[self.sol[-1], :, :]
        else:
            self.prev = self.prev | self.csim[self.sol[-1], :, :]
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev)
        

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, nsim):
        super(AdaptiveInfluence, self).__init__([v.index for v in g.vs])
        self.g = g
        self.p = p
        self.nsim = nsim
        self.h = h

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic(self.h, self.sol)
        csim = ic_sim_cond(self.h, self.p, self.nsim, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol)
        self.fsol = len(active) - LAMBDA*len(self.sol)

def test_graph():
    g = ig.Graph()
    g.add_vertices(7)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(4, 1)
    g.add_edge(4, 2)
    g.add_edge(4, 3)
    g.add_edge(5, 6)
    g.to_directed()
    return g

# Ratio of random greedy cardinality constraint
K_RATIO = 1

def compare_worker(i, g, pedge, nsim_ad, vrg_nonad):    
    print '-> worker', i, 'started.'
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, nsim_ad)
    (vrg_ad, _) = solver_ad.random_greedy(g.vcount()/K_RATIO)
    active_nonad = ic(h, vrg_nonad)
    active_ad = ic(h, vrg_ad)
    if DEBUG:
        print 'Non-adaptive: vs  =', vrg_nonad
        print '              val =', f_ic_base(h, vrg_nonad)
        print 'Adaptive:     vs  =', vrg_ad
        print '              val =', f_ic_base(h, vrg_ad)
    if f_ic_base(h, vrg_ad) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': len(vrg_ad),
            'f_nonad': f_ic_base(h, vrg_nonad),
            'f_ad': f_ic_base(h, vrg_ad)}

def compare(g, pedge, nsim_nonad, nsim_ad, niter, parallel=True, plot=False,
            savefig=False):
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.vs:
        st_nonad[v.index] = 0
        st_ad[v.index] = 0
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, pedge, nsim_nonad)
    (vrg_nonad, _) = solver_nonad.random_greedy(g.vcount()/K_RATIO)
    # Adaptive simulation
    arg = [g, pedge, nsim_ad, vrg_nonad]
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
        figname += '_NSIM_NONAD_' + str(nsim_nonad)
        figname += '_NSIM_AD_' + str(nsim_ad)
        figname += '_NITER_' + str(niter)
        if SAVEFIG:
            plt.savefig(os.path.abspath('../results/' + figname + '.pdf'),
                        orientation='landscape',
                        papertype='letter',
                        bbox_inches='tight',
                        format='pdf')
        plt.show()

def profile_aux():
    g = ig.Graph.Barabasi(100, 2)
    P_EDGE = 0.4
    NSIM_NONAD = 1000
    NSIM_AD = 1000
    NITER = 10
    PARALLEL = False
    PLOT = False
    SAVEFIG = False
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, PARALLEL, PLOT, SAVEFIG)

def profile():
    prof.run('influence.profile_aux()', sort='time')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Influence maximization')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug')
    args = parser.parse_args()
    DEBUG = args.debug

    random.seed(0)
    np.random.seed(0)
    tmp = nx.barabasi_albert_graph(200, 2)
    g = ig.Graph(directed=True)
    g.add_vertices(200)
    for u, v in tmp.edges_iter():
        g.add_edge(u, v)
        g.add_edge(v, u)
    P_EDGE = 0.4
    NSIM_NONAD = 10000
    NSIM_AD = 1
    NITER = 10
    PARALLEL = False
    PLOT = False
    SAVEFIG = False
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, PARALLEL, PLOT, SAVEFIG)

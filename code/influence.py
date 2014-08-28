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


def random_instance(g, p, rem=None, copy=False):
    if copy: g = g.copy()
    if rem == None: rem = g.edges()
    for e in rem:
        if random.random() > p:
            g.remove_edge(*e)
    return g

def ic(h, a):
    p = nx.all_pairs_shortest_path_length(h)
    active = set()
    for v in a:
        active = active | set(p[v].keys())
    return active

def ic_sim(g, p, niter, rem, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.number_of_nodes()
    csim = np.zeros(shape=(n, n, niter), dtype='bool')
    gedges = g.edges()
    for i in range(niter):
        g = random_instance(g, p, rem, copy=False)
        sp = nx.all_pairs_shortest_path_length(g)
        for v in g.nodes():
            csim[v, sp[v].keys(), i] = True
        g.add_edges_from(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_sim_cond(g, p, niter, rem, active, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.number_of_nodes()
    rest = set(g.nodes()) - set(active)
    csim = {v: 0 for v in rest}
    gedges = g.edges()
    for i in range(niter):
        g = random_instance(g, p, rem, copy=False)
        for v in rest:
            sp = nx.shortest_path_length(g, source=v)
            csim[v] += len(set(sp.keys()) | active)
        g.add_edges_from(rem)
        if pbar:
            pb.update(i)
    for v in csim:
        csim[v] /= (1.0*niter)
    if pbar:
        pb.finish()
    return csim

def get_live_dead(g, h, active):
    elive = set(h.edges(active))
    edead = set(g.edges(active)) - elive
    return (elive, edead)

def copy_without_edges(g, elist):
    h = g.copy()
    h.remove_edges_from(elist)
    return h

# Global weighting factor of node cost
LAMBDA = 1.0

def f_ic_base(h, a):
    active = ic(h, a)
    return len(active) - LAMBDA*len(a)

def f_ic(a, csim):
    a = list(set(a))
    if a == []: return 0
    act = csim[a[0], :, :]
    for v in a[1:]:
        act = act | csim[v, :, :]
    return (1.0*np.sum(act))/csim.shape[2] - LAMBDA*len(a)

def f_ic_ad(v, a, csim, active, fprev):
    a = set(a)
    active = list(set(active))
    if v in active:
        return fprev
    return csim[v] - LAMBDA*(len(a) + 1)

# vals is a dictionary from nodes (not necessarily all of them) to "strengths"
def draw_alpha(g, vals, pos=None, maxval=None):
    if not maxval: maxval = max(a.values())
    if not pos: pos = nx.spring_layout(g)
    nx.draw_networkx_edges(g, pos,
                           edge_color='#cccccc',
                           alpha=0.5,
                           arrows=False)
    for v in g.nodes():
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
    def __init__(self, g, p, nsim):
        super(NonAdaptiveInfluence, self).__init__(g.nodes())
        self.g = g
        self.p = p
        self.nsim = nsim

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        csim = ic_sim(self.g, self.p, self.nsim, rem=self.g.edges(), pbar=True)
        self.f = lambda v, a: f_ic(a + [v], csim)

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, nsim):
        super(AdaptiveInfluence, self).__init__(g.nodes())
        self.g = g
        self.p = p
        self.nsim = nsim
        self.h = h

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic(self.h, self.sol)
        (elive, edead) = get_live_dead(self.g, self.h, active)
        r = copy_without_edges(self.g, edead)
        rem = set(r.edges()) - elive
        csim = ic_sim_cond(r, self.p, self.nsim, rem=rem, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol)
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

def compare_worker(i, g, pedge, nsim_ad, vrg_nonad):    
    print '-> worker', i, 'started.'
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, nsim_ad)
    (vrg_ad, _) = solver_ad.random_greedy(g.number_of_nodes()/K_RATIO)
    active_nonad = ic(h, vrg_nonad)
    active_ad = ic(h, vrg_ad)
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

def compare(g, pedge, nsim_nonad, nsim_ad, niter, parallel=True, plot=False):
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.nodes():
        st_nonad[v] = 0
        st_ad[v] = 0
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, pedge, nsim_nonad)
    (vrg_nonad, _) = solver_nonad.random_greedy(g.number_of_nodes()/K_RATIO)
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
        plt.savefig(os.path.abspath('../results/' + figname + '.pdf'),
                    orientation='landscape',
                    papertype='letter',
                    bbox_inches='tight',
                    format='pdf')
        plt.show()

def profile_aux():
    g = nx.barabasi_albert_graph(50, 2)
    P_EDGE = 0.4
    NSIM_NONAD = 1000
    NSIM_AD = 1000
    NSIM_AP_AD = 1000
    NITER = 10
    PARALLEL = False
    PLOT = False
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, PARALLEL, PLOT)

def profile():
    prof.run('influence.profile_aux()', sort='time')

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    g = nx.barabasi_albert_graph(100, 2)
    P_EDGE = 0.4
    NSIM_NONAD = 1000
    NSIM_AD = 1000
    NITER = 10
    PARALLEL = False
    PLOT = False
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, PARALLEL, PLOT)

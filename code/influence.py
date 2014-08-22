import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import submod


def random_instance(g, p, prz=(set(), set()), seed=None):
    h = g.copy()
    to_remove = []
    if seed: random.seed(seed)
    elive, edead = prz
    for e in h.edges_iter():
        if e in elive:
            continue
        elif e in edead:
            to_remove.append(e)
        elif random.random() > p:
            to_remove.append(e)
    h.remove_edges_from(to_remove)
    return h

def independent_cascade(h, a):
    p = nx.all_pairs_shortest_path_length(h)
    active = set()
    for v in a:
        active = active | set(p[v].keys())
    return active

def cascade_sim(g, p, niter, prz=None):
    csim = []
    for i in range(niter):
        if prz:
            h = random_instance(g, p, prz)
        else:
            h = random_instance(g, p)
        sp = nx.all_pairs_shortest_path_length(h)
        tmp = {}
        for v in h.nodes():
            tmp[v] = set(sp[v].keys())
        csim.append(tmp)
    return csim

def count_activations(a, csim):
    vact = {}
    for d in csim:
        tmp = set()
        for v in a:
            tmp = tmp | d[v]
        for u in tmp:
            try:
                vact[u] += 1
            except KeyError:
                vact[u] = 1
    return vact

def finf_base(h, a):
    active = independent_cascade(h, a)
    return len(active) - 1*len(a)

# TODO: Refactor to use finf_base
def finf(a, csim):
    a = set(a)
    nact = []
    for d in csim:
        tmp = set()
        for v in a:
            tmp = tmp | d[v]
        nact.append(len(tmp))
    return np.mean(nact) - 1*len(a)

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
        self.csim = []

    def init_f_hook(self):
#        print 'Running cascade simulation...',
        sys.stdout.flush()
        self.csim = cascade_sim(self.g, self.p, self.nsim)
#        print 'completed'
        self.f = lambda a: finf(a, self.csim)

    def draw_influence(self, a):
        dact = count_activations(a, self.csim)
        draw_alpha(self.g, dact, maxval=self.nsim)
        plt.show()

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, nsim):
        super(AdaptiveInfluence, self).__init__(g.nodes())
        self.g = g
        self.p = p
        self.nsim = nsim
        self.h = h

    def init_f_hook(self):
        self.update_f_hook()

    def update_f_hook(self):
        active = independent_cascade(self.h, self.sol)
        elive = set(self.h.edges(active))
        edead = set(self.g.edges(active)) - elive
#        print 'Running cascade simulation', str(len(self.sol)) + '...',
        sys.stdout.flush()
        csim = cascade_sim(self.g, self.p, self.nsim, prz=(elive, edead))
#        print '-------------'
#        print csim
#        print '-------------'
#        print 'completed'
        self.f = lambda a: finf(a, csim)
        self.fsol = self.f(self.sol)

def test_graph():
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(5, 2)
    g.add_edge(5, 3)
    g.add_edge(5, 4)
    g.add_edge(6, 7)
    return g.to_directed()

def run_non_adaptive(draw=True):
    #g = nx.watts_strogatz_graph(100, 4, 0.5)
    g = nx.barabasi_albert_graph(100, 2)
    #g = test_graph()
    P_EDGE = 0.3
    NSIM = 1000

    solver = NonAdaptiveInfluence(g, P_EDGE, NSIM)
    (vrg, frg) = solver.random_greedy(len(g.nodes()))
    print 'RG =', frg, vrg

    if draw:
        solver.draw_influence(vrg)
    
# TODO: Make it run on same instances as non-adaptive
def run_adaptive():
    g = nx.barabasi_albert_graph(100, 2)
    #g = test_graph()
    P_EDGE = 0.3
    NSIM = 1000

    vals = []
    for i in range(100):
        print '------> i =', i
        h = random_instance(g, P_EDGE)
        solver = AdaptiveInfluence(g, h, P_EDGE, NSIM)
        (vrg, frg) = solver.random_greedy(len(g.nodes()))
        print 'frg =', frg
        print (vrg, finf_base(h, vrg))
        vals.append(finf_base(h, vrg))
    print 'favg =', np.mean(vals)
#        nx.draw_networkx(h)
#        plt.show()

def compare_worker(i, g, p_edge, nsim_ad, vrg_nonad):
    print 'worker', i, 'started.'
    h = random_instance(g, p_edge)
    solver_ad = AdaptiveInfluence(g, h, p_edge, nsim_ad)
    (vrg_ad, _) = solver_ad.random_greedy(len(g.nodes()))
    active_nonad = independent_cascade(h, vrg_nonad)
    active_ad = independent_cascade(h, vrg_ad)
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': len(vrg_ad),
            'f_nonad': finf_base(h, vrg_nonad),
            'f_ad': finf_base(h, vrg_ad)}

def compare():
    #g = test_graph()
    g = nx.barabasi_albert_graph(50, 2)
    P_EDGE = 0.4
    NSIM_NONAD = 10000
    NSIM_AD = 1000
    NITER = 1
    PARALLEL = False
    PLOT = False
    # Init
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.nodes():
        st_nonad[v] = 0
        st_ad[v] = 0
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, P_EDGE, NSIM_NONAD)
    (vrg_nonad, _) = solver_nonad.random_greedy(len(g.nodes()))
    # Adaptive simulation
    arg = [g, P_EDGE, NSIM_AD, vrg_nonad]
    if PARALLEL:
        res = joblib.Parallel(n_jobs=4)((compare_worker, [i] + arg, {})
                                        for i in range(NITER))
    else:
        res = [compare_worker(*([i] + arg)) for i in range(NITER)]
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
    if PLOT:
        _, (ax1, ax2) = plt.subplots(1, 2)
        plt.gcf().set_size_inches(16, 9)
        plt.sca(ax1)
        ax1.set_aspect('equal')
        draw_alpha(g, st_nonad, pos=pos, maxval=NITER)
        plt.sca(ax2)
        ax2.set_aspect('equal')
        draw_alpha(g, st_ad, pos=pos, maxval=NITER)
        figname = 'INF'
        figname += '_p_edge_' + str(P_EDGE*100)
        figname += '_nsim_nonad_' + str(NSIM_NONAD)
        figname += '_NSIM_AD_' + str(NSIM_AD)
        figname += '_NITER_' + str(NITER)
        plt.savefig(os.path.abspath('../results/' + figname + '.pdf'),
                    orientation='landscape',
                    papertype='letter',
                    bbox_inches='tight',
                    format='pdf')
        plt.show()

if __name__ == "__main__":
    compare()

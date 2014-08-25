import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import bitarray as ba
import progressbar
import submod


def random_instance(g, p, rem=None, copy=False):
    if copy: g = g.copy()
    if rem == None: rem = g.edges()
    for e in rem:
        if random.random() > p:
            g.remove_edge(*e)
    return g

def independent_cascade(h, a):
    p = nx.all_pairs_shortest_path_length(h)
    active = set()
    for v in a:
        active = active | set(p[v].keys())
    return active

def cascade_sim(g, p, niter, rem=None, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    csim = []
    gnodes = g.nodes()
    gedges = g.edges()
    for i in range(niter):
        g = random_instance(g, p, rem, copy=False)
        sp = nx.all_pairs_shortest_path_length(g)
        tmp = {}
        for v in g.nodes():
            b = g.number_of_nodes()*ba.bitarray('0')
            for u in sp[v]:
                b[u] = True
            tmp[v] = b
        csim.append(tmp)
        g.clear()
        g.add_nodes_from(gnodes)
        g.add_edges_from(gedges)
        if pbar:
            pb.update(i)
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

def finf_base(h, a):
    active = independent_cascade(h, a)
    return len(active) - 1.0*len(a)

def finf(a, csim):
    a = set(a)
    nact = 0
    for d in csim:
        tmp = None
        for v in a:
            if not tmp:
                tmp = d[v]
            else:
                tmp = tmp | d[v]
        if tmp:
            nact += tmp.count(True)
    return (1.0*nact)/len(csim) - 1.0*len(a)

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

    # TODO: Remove 'self' from csim
    def init_f_hook(self):
        sys.stdout.flush()
        self.csim = cascade_sim(self.g, self.p, self.nsim, pbar=True)
        self.f = lambda a: finf(a, self.csim)

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
        (elive, edead) = get_live_dead(self.g, self.h, active)
        r = copy_without_edges(self.g, edead)
        rem = set(r.edges()) - elive
        csim = cascade_sim(r, self.p, self.nsim, rem=rem)
        self.f = lambda a: finf(a, csim)
        self.fsol = self.f(self.sol)

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

def compare_worker(i, g, p_edge, nsim_ad, vrg_nonad):
    print '-> worker', i, 'started.'
    h = random_instance(g, p_edge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, p_edge, nsim_ad)
    (vrg_ad, _) = solver_ad.random_greedy(len(g.nodes()))
    active_nonad = independent_cascade(h, vrg_nonad)
    active_ad = independent_cascade(h, vrg_ad)
    eval1 = finf_base(h, vrg_ad)
    eval2 = solver_ad.fsol
    print 'eval1:', eval1
    print 'eval2:', eval2
    if eval1 != eval2: raise 'Inconsistent adaptive function values'
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': len(vrg_ad),
            'f_nonad': finf_base(h, vrg_nonad),
            'f_ad': finf_base(h, vrg_ad)}

def compare():
    #g = test_graph()
    g = nx.barabasi_albert_graph(100, 2)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    P_EDGE = 0.4
    NSIM_NONAD = 10000
    NSIM_AD = 1000
    NITER = 100
    PARALLEL = True
    PLOT = True
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

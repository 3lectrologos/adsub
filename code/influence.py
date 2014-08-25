import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
import joblib
import bitarray as ba
import progressbar
import submod
import util


MAX_DIST = np.iinfo('i').max

def random_instance(g, p, prz=(set(), set()), seed=None):
    if seed: random.seed(seed)
    pm = g.new_edge_property('bool')
    elive, edead = prz
    for e in g.edges():
        ne = util.e(e)
        if ne in elive:
            pm[e] = True
        elif ne in edead:
            pm[e] = False
        elif random.random() > p:
            pm[e] = False
        else:
            pm[e] = True
    h = gt.GraphView(g, efilt=pm)
    return h

def independent_cascade(h, a):
    sp = gt.shortest_distance(h)
    active = set()
    for vn in a:
        v = h.vertex(vn)
        new = set(np.arange(h.num_vertices())[np.array(sp[v]) < MAX_DIST])
        active = active | new
    return active

def cascade_sim(g, p, niter, prz=None, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    csim = []
    for i in range(niter):
        if prz:
            h = random_instance(g, p, prz)
        else:
            h = random_instance(g, p)
        sp = gt.shortest_distance(h)
        tmp = {}
        for v in h.vertices():
            tmp[g.vertex_index[v]] = ba.bitarray(
                list(np.array(sp[v]) < MAX_DIST))
        csim.append(tmp)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def get_live_dead(g, h, active):
    active = set(active)
    elive = set()
    edead = set()
    for v in active:
        for e in g.vertex(v).out_edges():
            ne = util.e(e)
            if h.edge(*ne):
                elive.add(ne)
            else:
                edead.add(ne)
    return (elive, edead)

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
    if not pos: pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
#    nx.draw_networkx_edges(g, pos,
#                           edge_color='#cccccc',
#                           alpha=0.5,
#                           arrows=False)
    vcol = g.new_vertex_property('vector<float>')
    for v in g.vertices():
        if v not in vals or vals[v] == 0:
            vcol[v] = [0.3, 0.3, 0.3, 1]
        else:
            vcol[v] = [0, 0, 1, (1.0*vals[v])/maxval]
            #alpha=(1.0*vals[v])/maxval
    gt.graph_draw(g, pos, vertex_fill_color=vcol)
#                            font_size=10,
#                            font_color='#eeeeee')

class BaseInfluence(submod.AdaptiveMax):
    pass

class NonAdaptiveInfluence(BaseInfluence):
    def __init__(self, g, p, nsim):
        vlabels = [g.vertex_index[v] for v in g.vertices()]
        super(NonAdaptiveInfluence, self).__init__(vlabels)
        self.g = g
        self.p = p
        self.nsim = nsim
        self.csim = []

    def init_f_hook(self):
#        print 'Running cascade simulation...',
        sys.stdout.flush()
        self.csim = cascade_sim(self.g, self.p, self.nsim, pbar=True)
#        print 'completed'
        self.f = lambda a: finf(a, self.csim)

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, nsim):
        vlabels = [g.vertex_index[v] for v in g.vertices()]
        super(AdaptiveInfluence, self).__init__(vlabels)
        self.g = g
        self.p = p
        self.nsim = nsim
        self.h = h

    def init_f_hook(self):
        self.update_f_hook()

    def update_f_hook(self):
        #print 'sol =', self.sol
        active = independent_cascade(self.h, self.sol)
        #print 'active =', active
        elive, edead = get_live_dead(self.g, self.h, active)
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
    g = gt.Graph()
    es = [(0, 1), (0, 2), (0, 3), (4, 1), (4, 2), (4, 3), (5, 6),
          (1, 0), (2, 0), (3, 0), (1, 4), (2, 4), (3, 4), (6, 5)]
    g.add_edge_list(es)
    g.set_fast_edge_removal(True)
    return g

def compare_worker(i, g, p_edge, nsim_ad, vrg_nonad):
    print 'worker', i, 'started.'
    h = random_instance(g, p_edge)
    solver_ad = AdaptiveInfluence(g, h, p_edge, nsim_ad)
    (vrg_ad, _) = solver_ad.random_greedy(g.num_vertices())
    active_nonad = independent_cascade(h, vrg_nonad)
    active_ad = independent_cascade(h, vrg_ad)
    print 'eval1:', finf_base(h, vrg_ad)
    print 'eval2:', solver_ad.fsol
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': len(vrg_ad),
            'f_nonad': finf_base(h, vrg_nonad),
            'f_ad': finf_base(h, vrg_ad)}

def compare():
    #g = test_graph()
    g = gt.price_network(50, 2, directed=False)
    for e in g.edges():
        g.add_edge(e.target(), e.source())
    g.set_directed(True)
    g.set_fast_edge_removal(True)

    P_EDGE = 0.4
    NSIM_NONAD = 1000
    NSIM_AD = 100
    NITER = 10
    PARALLEL = False
    PLOT = False
    # Init
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.vertices():
        st_nonad[v] = 0
        st_ad[v] = 0
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, P_EDGE, NSIM_NONAD)
    (vrg_nonad, _) = solver_nonad.random_greedy(g.num_vertices())
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
            st_nonad[g.vertex(v)] += 1
        for v in r['active_ad']:
            st_ad[g.vertex(v)] += 1
    # Print results
    print 'Non-adaptive | favg =', np.mean([r['f_nonad'] for r in res]),
    print ',     #nodes =', len(vrg_nonad)
    print 'Adaptive     | favg =', np.mean([r['f_ad'] for r in res]),
    print ', avg #nodes =', np.mean([r['v_ad'] for r in res])
    # Plotting
    if PLOT:
        pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
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

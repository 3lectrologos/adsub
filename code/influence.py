import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import submod


def random_instance(g, p, prz=(set(), []), seed=None):
    h = g.copy()
    to_remove = []
    if seed: random.seed(seed)
    vprz, eprz = prz
    for e in h.edges_iter():
        if e[0] in vprz:
            if e not in eprz:
                to_remove.append(e)
            continue
        if random.random() > p:
            to_remove.append(e)
    h.remove_edges_from(to_remove)
    return h

def independent_cascade(h, a):
    p = nx.all_pairs_shortest_path_length(h)
    active = set()
    for v in a:
        active = active | set(p[v].keys())
    return active

def cascade_sim(g, p, niter):
    s = []
    for i in range(niter):
        h = random_instance(g, p)
        sp = nx.all_pairs_shortest_path_length(h)
        tmp = {}
        for v in g.nodes():
            tmp[v] = set(sp[v].keys())
        s.append(tmp)
    return s

def count_activations(a, s):
    vact = {}
    for d in s:
        tmp = set()
        for v in a:
            tmp = tmp | d[v]
        for u in tmp:
            try:
                vact[u] += 1
            except KeyError:
                vact[u] = 1
    return vact

def draw_influence(g, a, csim, nsim):
    pos = nx.spring_layout(g)
    nx.draw_networkx_edges(g, pos,
                           edge_color='#cccccc', alpha=0.5, arrows=False)
    dact = count_activations(vrg, csim)
    vact = set(dact.keys())
    vnorm = set(g.nodes()) - vact
    nx.draw_networkx_nodes(g, pos, nodelist=vnorm,
                           node_color='#555555', alpha=0.5)
    for v in vact:
        nx.draw_networkx_nodes(g, pos, nodelist=[v],
                               node_color='b', alpha=(1.0*dact[v])/nsim)
    nx.draw_networkx_labels(g, pos, nodelist=[v],
                            font_size=10, font_color='#eeeeee')
    plt.show()

class NonAdaptiveInfluence(submod.AdaptiveMax):
    def finf(self, a):
        a = set(a)
        nact = []
        for d in self.csim:
            tmp = set()
            for v in a:
                tmp = tmp | d[v]
            nact.append(len(tmp))
        return np.mean(nact) - 1*len(a)

    def init_f_hook(self):
        print 'Running cascade simulation...',
        sys.stdout.flush()
        self.csim = cascade_sim(data['g'], data['p'], data['nsim'])
        print 'completed'
        self.f = lambda a: self.finf(a)

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

if __name__ == "__main__":
    #g = nx.watts_strogatz_graph(100, 4, 0.5)
    #g = nx.barabasi_albert_graph(100, 2)
    g = test_graph()

    P_EDGE = 0.3
    NSIM = 1000
    data = {'g': g, 'p': P_EDGE, 'nsim': NSIM}
    solver = NonAdaptiveInfluence(g.nodes(), data=data)
    (vrg, farg) = solver.random_greedy(len(g.nodes()))
    print 'RG =', farg, vrg

    if True:
        draw_influence(g, vrg, solver.csim, NSIM)

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import submod


def f_influence(g, a, p=0.2, niter=1000):
    rs = []
    for i in range(niter):
        r = independent_cascade(g, a, p)
        rs.append(len(r))
    return (1.0*sum(rs))/len(rs)

def random_instance(g, p, seed=None):
    h = g.copy()
    to_remove = []
    if seed: random.seed(seed)
    for e in h.edges_iter():
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

def test_graph():
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(1, 5)
    g.add_edge(5, 2)
    g.add_edge(5, 3)
    g.add_edge(5, 4)
    g.add_edge(6, 7)
    return g.to_directed()

def cascade_sim(g, p, niter):
    s = {}
    for v in g.nodes():
        s[v] = []
    for i in range(niter):
        h = random_instance(g, p)
        sp = nx.all_pairs_shortest_path_length(h)
        for v in g.nodes():
            s[v].append(sp[v].keys())
    return s


if __name__ == "__main__":
    #g = nx.watts_strogatz_graph(50, 4, 0.5)
    g = nx.barabasi_albert_graph(100, 3, 0)
    #g = test_graph()
    
    if True:
        pos = nx.spring_layout(g)
        nx.draw(g, pos)
    #    nx.draw_networkx_nodes(g, pos, nodelist=active, node_color='b')
        plt.show()

    s = cascade_sim(g, 0.2, 1000)
    for v in g.nodes():
        print v, np.mean([len(a) for a in s[v]])

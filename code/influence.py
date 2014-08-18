import random
import Queue
import matplotlib.pyplot as plt
import networkx as nx
import submod


def adaptive_max(g, k):
    pass

def f(v):
    v = set(v)
    if v == []:
        return 0
    if v == set([1, 2, 3]):
        return 2.1
    if v == set([1]):
        return 1.1
    if v == set([2]):
        return 1
    if v == set([3]):
        return 1
    if v == set([1, 2]):
        return 1.5
    if v == set([1, 3]):
        return 2.1
    if v == set([2, 3]):
        return 2

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

def independent_cascade(g, a, p):
    h = random_instance(g, p)
    active = set(a)
    previous = set(a)
    while True:
        for v in previous:
            active = active | set(nx.all_neighbors(h, v))
        if len(active) == len(previous):
            break
        else:
            previous = active
    return active

#g = nx.watts_strogatz_graph(50, 4, 0.5)
#g = nx.barabasi_albert_graph(50, 7, 0)
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
g = g.to_directed()

if True:
    pos = nx.spring_layout(g)
    nx.draw(g, pos)
#    nx.draw_networkx_nodes(g, pos, nodelist=active, node_color='b')
    plt.show()

print submod.greedy_max(lambda a: f_influence(g, a), set(g.nodes()), 2)

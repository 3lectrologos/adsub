import networkx as nx
import matplotlib.pyplot as plt
import submod


def f_wavg(g, a, p=0.7):
    f = 0
    for u in a:
        for u, v in g.edges(u):
            w = g.edge[u][v]['w']
            if v not in a:
                f += p*w
    return f

g = nx.Graph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_edge(1, 2, w=1)
g.add_edge(1, 3, w=2)

print f_wavg(g, [1])
print f_wavg(g, [2])
print f_wavg(g, [3])
print f_wavg(g, [1, 2])
print f_wavg(g, [1, 2, 3])
print 'max = ', submod.greedy_max(lambda a: f_wavg(g, a), g.nodes(), 2)

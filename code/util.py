import csv
import networkx as nx
import igraph as ig
import influence


def read_graph(filename):
    g = nx.DiGraph()
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        i = 0
        for row in reader:
            if row[0][0] == '#':
                continue
            else:
                v = int(row[0])
                u = int(row[1])
                g.add_edge(u, v)
    g = nx.convert_node_labels_to_integers(g, ordering='decreasing degree')
    h = ig.Graph(directed=True)
    h.add_vertices(g.number_of_nodes())
    h.add_edges(g.edges())
    return h

def save_graph(g, fout):
    with open(fout, 'wb') as f:
        for e in g.es:
            f.write(str(e.source) + ' ' + str(e.target) + '\n')


def save_subgraph(model, n):
    (name, g) = influence.get_tc(model, n)
    save_graph(g, name + '.txt')

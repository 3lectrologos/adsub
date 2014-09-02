import csv
import networkx as nx
import igraph as ig


def read_snap_graph(filename):
    g = nx.DiGraph()
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
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

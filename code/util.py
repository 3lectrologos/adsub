import os
import csv
import subprocess
import numpy as np
import matplotlib.pylab as plt
import networkx as nx
import igraph as ig
import influence


DIR_DATA = os.path.abspath('../data/')
DIR_RES = os.path.abspath('../results/')
DIR_TEMPLATE = os.path.abspath('../templates/')

def read_graph(filename, directed=True, nxgraph=False):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0][0] == '#':
                continue
            else:
                u = int(row[0])
                v = int(row[1])
                g.add_edge(u, v)
    g = nx.convert_node_labels_to_integers(g, ordering='decreasing degree')
    if nxgraph:
        return g
    h = ig.Graph(directed=directed)
    h.add_vertices(g.number_of_nodes())
    h.add_edges(g.edges())
    return h

def read_sig_graph(filename):
    g = nx.Graph()
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        i = 0
        for row in reader:
            u = str(row[0])
            v = str(row[5])
            g.add_edge(u, v)
    g = nx.convert_node_labels_to_integers(g, ordering='decreasing degree')
    h = ig.Graph(directed=False)
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

def get_tc(fname, n=None, m=None, directed=True, nxgraph=False):
    if fname == 'B_A':
        if n == None: n = 1000
        if m == None: m = 2
        name = 'B_A_{0}_{1}'.format(n, m)
        g = ig.Graph.Barabasi(n, m)
        if directed:
            g.to_directed()
        else:
            g.to_undirected()
        for v in g.vs:
            v['i'] = v.index
    else:
        g = read_graph(os.path.join(DIR_DATA, fname + '.txt'), directed, nxgraph)
        g.simplify()
        if n != None:
#            g = sample_RDN(g, n)
            g = sample_RJ(g, n)
        for v in g.vs:
            v['i'] = v.index
        name = fname + '_' + str(len(g.vs))
    return (name, g)

def sample_RDN(g, n):
    if n == None:
        return g
    pr = np.array(g.degree(g.vs), dtype=np.double)
    pr /= np.sum(pr)
    new_vs = np.random.choice(g.vs, size=n, replace=False, p=pr)
    return g.subgraph(new_vs)

def sample_RJ(g, n):
    if n == None:
        return g
    print n
    JP = 0.15
    v = np.random.choice(g.vs)
    new_vs = set()
    while len(new_vs) < n:
        print len(new_vs)
        new_vs.add(v)
        nbs = g.neighbors(v)
        if len(nbs) == 0 or np.random.random < JP:
            v = np.random.choice(g.vs)
        else:
            v = np.random.choice(nbs)
    print 'new_vs =', len(new_vs)
    return g.subgraph(new_vs)

def plot_degree_dist(g):
    xs, ys = zip(*[(left, count) for left, _, count in 
                   g.degree_distribution().bins()])
    plt.loglog(xs, ys, 'o')
    plt.show()

def get_coords(x, y):
    s = ''
    z = sorted(zip(x, y))
    for e in z:
        s += '(' + str(e[0]) + ',' + str(e[1]) + ')'
    return s
    
def replace(datadir, fname, d):
    with open(os.path.join(DIR_TEMPLATE, fname), 'r') as fin:
        s = fin.read()
        for k in d:
            s = s.replace('%' + k + '%', d[k])
            with open(os.path.join(datadir, fname), 'w') as fout:
                fout.write(s)

def maketex(datadir, fname):
    cwd = os.getcwd()
    os.chdir(datadir)
    subprocess.call(['latexmk', '-pdf', fname])
    subprocess.call(['latexmk', '-c'])
    os.chdir(cwd)

def clean(datadir):
    cwd = os.getcwd()
    os.chdir(datadir)
    subprocess.call(['latexmk', '-C'])
    subprocess.call('rm -f *.tex', shell=True)
    os.chdir(cwd)

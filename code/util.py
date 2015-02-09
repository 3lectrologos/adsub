import os
import csv
import subprocess
import numpy as np
import matplotlib.pylab as plt
import networkx as nx
import igraph as ig


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DIR_DATA = os.path.abspath(os.path.join(THIS_DIR, '../data'))
DIR_RES = os.path.abspath(os.path.join(THIS_DIR, '../results'))
DIR_TEMPLATE = os.path.abspath(os.path.join(THIS_DIR, '../templates'))


def read_graph(filename, directed=False, nxgraph=False):
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


def get_tc(fname, n, directed=True, nxgraph=False):
    if fname == 'B_A':
        g = ig.Graph.Barabasi(n, 2)
        if directed:
            g.to_directed()
        else:
            g.to_undirected()
        for v in g.vs:
            v['i'] = v.index
    elif fname == 'E_R':
        g = ig.Graph.Erdos_Renyi(n, m=2000)
        if directed:
            g.to_directed()
        else:
            g.to_undirected()
        for v in g.vs:
            v['i'] = v.index
    elif fname == 'W_S':
        g = ig.Graph.Watts_Strogatz(1, 1000, 3, 0.1)
        if directed:
            g.to_directed()
        else:
            g.to_undirected()
        for v in g.vs:
            v['i'] = v.index
    else:
        g = read_graph(os.path.join(DIR_DATA, fname + '.txt'), directed, nxgraph)
        g.simplify()
        if n != None and n < g.vcount():
            g = sample_RW(g, n)
        print 'check:', g.vcount()
        for v in g.vs:
            v['i'] = v.index
            print v.index
    name = fname + '_' + str(len(g.vs))
    return (name, g)


def sample_RDN(g, n):
    if n == None:
        return g
    pr = np.array(g.degree(g.vs), dtype=np.double)
    pr /= np.sum(pr)
    new_vs = np.random.choice(g.vs, size=n, replace=False, p=pr)
    return g.subgraph(new_vs)


def sample_RW_edges(g, n):
    MAX_COUNTER = 100 * n
    if n == None:
        return g
    RP = 0.15
    vstart = np.random.choice(g.vs).index    
    v = vstart
    new_vs = set()
    new_es = set()
    new_e = None
    counter = 0
    while len(new_vs) < n:
        new_vs.add(v)
        if new_e is not None:
            new_es.add(new_e)
            new_e = None
        nbs = g.neighbors(v)
        if len(nbs) == 0 or np.random.random() < RP:
            v = vstart
            print '--->', v
        else:
            w = np.random.choice(nbs)
            new_e = (v, w)
            v = w
        print len(new_vs)
        if counter >= MAX_COUNTER:
            vstart = np.random.choice(g.vs).index
            v = vstart
            counter = 0
            print '==============> resetting'
        else:
            counter += 1
        print len(new_vs)
    print 'vcount =', len(new_vs)
    return g.subgraph_edges(new_es)


def sample_RW(g, n):
    MAX_COUNTER = 100 * n
    if n == None:
        return g
    RP = 0.15
    vstart = np.random.choice(g.vs).index
    v = vstart
    new_vs = set()
    counter = 0
    while len(new_vs) < n:
        new_vs.add(v)
        nbs = g.neighbors(v)
        if len(nbs) == 0 or np.random.random() < RP:
            v = vstart
            print '--->', v
        else:
            v = np.random.choice(nbs)
        print len(new_vs)
        if counter >= MAX_COUNTER:
            vstart = np.random.choice(g.vs).index
            v = vstart
            counter = 0
            print '==============> resetting'
        else:
            counter += 1
        print len(new_vs)
    print 'vcount =', len(new_vs)
    return g.induced_subgraph(new_vs)


def sample_RJ(g, n):
    if n == None:
        return g
    JP = 0.15
    v = np.random.choice(g.vs).index
    new_vs = set()
    while len(new_vs) < n:
        new_vs.add(v)
        nbs = g.neighbors(v)
        if len(nbs) == 0 or np.random.random() < JP:
            v = np.random.choice(g.vs).index
            print '--->', v
        else:
            v = np.random.choice(nbs)
        print len(new_vs)
    print 'vcount =', len(new_vs)
    return g.induced_subgraph(new_vs)


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

    
def replace(datadir, fname, d, outname=None):
    if outname is None:
        outname = fname
    with open(os.path.join(DIR_TEMPLATE, fname), 'r') as fin:
        s = fin.read()
        for k in d:
            s = s.replace('%' + k + '%', d[k])
            with open(os.path.join(datadir, outname), 'w') as fout:
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

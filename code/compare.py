import time
import networkx as nx
import igraph as ig
import graph_tool.all as gt
import progressbar


def measure_time(name, f, n, niter):
    (start, end) = f(n, niter)
    print name, ':\t', end - start

def run_nx(n, niter):
    pb = progressbar.ProgressBar(maxval=niter).start()
    g = nx.barabasi_albert_graph(n, 2)
    start = time.time()
    for i in range(niter):
        nx.all_pairs_shortest_path_length(g)
        pb.update(i)
    pb.finish()
    end = time.time()
    return (start, end)

def run_igraph(n, niter):
    pb = progressbar.ProgressBar(maxval=niter).start()
    g = ig.Graph.Barabasi(n, 2)
    start = time.time()
    for i in range(niter):
        g.shortest_paths()
        pb.update(i)
    pb.finish()
    end = time.time()
    return (start, end)

def run_igraph_comp(n, niter):
    pb = progressbar.ProgressBar(maxval=niter).start()
    g = ig.Graph.Barabasi(n, 2)
    start = time.time()
    for i in range(niter):
        for v in g.vs:
            g.subcomponent(v, mode=ig.OUT)
        pb.update(i)
    pb.finish()
    end = time.time()
    return (start, end)

def run_graph_tool(n, niter):
    pb = progressbar.ProgressBar(maxval=niter).start()
    g = gt.price_network(n, 2, directed=False)
    for e in g.edges():
        g.add_edge(e.target(), e.source())
    g.set_directed(True)
    start = time.time()
    for i in range(niter):
        gt.shortest_distance(g)
        pb.update(i)
    pb.finish()
    end = time.time()
    return (start, end)

if __name__ == "__main__":
    N = 5000
    NITER = 1
    measure_time('networkx', run_nx, N, NITER)
    measure_time('igraph', run_igraph, N, NITER)
    measure_time('igraph_comp', run_igraph_comp, N, NITER)
    measure_time('graph_tool', run_graph_tool, N, NITER)

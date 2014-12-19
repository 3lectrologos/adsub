import argparse
import random
import os
import sys
import time
import calendar as cal
import cPickle as pcl
import cProfile as prof
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx
import submod
import influence
import util


def cutval(g, cut):
    cut = set(cut)
    rest = set(g.vs.indices) - cut
    return len(g.es.select(_between=(cut, rest)))

def cutdif(g, u, cut, cut_edges):
    if u in cut:
        return (0, set(), set())
    cut_edges = set(cut_edges)
    u_edges = set(g.incident(u))
    added_edges = u_edges - cut_edges
    removed_edges = cut_edges & u_edges
    return (len(added_edges) - len(removed_edges), added_edges, removed_edges)

def pcut(g, cset, cut, cut_edges):
    cut_edges = set(cut_edges)
    dif = []
    for u in cset:
        (diflen, _, _) = cutdif(g, u, cut, cut_edges)
        dif.append(diflen)
    return np.mean(dif)

class NonAdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, nsim):
        super(NonAdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.ins = [(set(), set(), 0)] * nsim

    def mean_pcut(self, cid):
        return np.mean([pcut(self.g, self.csets[cid], cut, cut_edges) + val
                        for (cut, cut_edges, val) in self.ins])

    def init_f_hook(self):
        super(NonAdaptiveMaxCut, self).init_f_hook()
        self.f = lambda v, a: self.mean_pcut(v)

    def update_f_hook(self):
        print 'nonad:', len(self.sol)
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        cset = self.csets[self.sol[-1]]
        for i, (cut, cut_edges, val) in enumerate(self.ins):
            vcut = np.random.choice(cset)
            elen, eadd, erem = cutdif(self.g, vcut, cut, cut_edges)
            new_val = val + elen
            new_cut = cut | set([vcut])
            new_cut_edges = (cut_edges - erem) | eadd
            self.ins[i] = (new_cut, new_cut_edges, new_val)

class AdaptiveMaxCut(submod.AdaptiveMax):
    def __init__(self, g, csets, instance):
        super(AdaptiveMaxCut, self).__init__(csets.keys())
        self.g = g
        self.csets = csets
        self.instance = instance

    def init_f_hook(self):
        super(AdaptiveMaxCut, self).init_f_hook()
        self.cut = set()
        self.cut_edges = set()
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut, self.cut_edges)

    def update_f_hook(self):
        vcut = self.instance[self.sol[-1]]
        elen, eadd, erem = cutdif(self.g, vcut, self.cut, self.cut_edges)
        self.fsol += elen
        self.cut.add(vcut)
        self.cut_edges = (self.cut_edges - erem) | eadd
        self.f = lambda v, a: pcut(self.g, self.csets[v], self.cut, self.cut_edges) + self.fsol

def random_instance(csets):
    inst = {}
    for cid in csets:
        inst[cid] = np.random.choice(csets[cid])
    return inst

def eval_instance(g, inst, cutids):
    cut = [inst[cid] for cid in cutids]
    return cutval(g, cut)

def compare(g, csets, nsim_nonad, niter, k):
    f_nonad = []
    f_ad = []
    solver_nonad = NonAdaptiveMaxCut(g, csets, nsim_nonad)
    cut_nonad, _ = solver_nonad.random_greedy(k)
    for i in range(niter):
        #print 'i =', i
        instance = random_instance(csets)
        solver_ad = AdaptiveMaxCut(g, csets, instance)
        cut_ad, _ = solver_ad.random_greedy(k)
        f_nonad.append(eval_instance(g, instance, cut_nonad))
        f_ad.append(eval_instance(g, instance, cut_ad))
    fm_nonad = np.mean(f_nonad)
    fm_ad = np.mean(f_ad)
    imp = 100.0 * (fm_ad - fm_nonad) / fm_nonad
    print 'f_nonad =', fm_nonad
    print 'f_ad =', fm_ad
    print 'improvement = {0:.1}'.format(imp)
    return {
        'f_nonad': fm_nonad,
        'f_ad': fm_ad,
        'imp': imp
        }

def test_graph():
    g = ig.Graph()
    g.add_vertices(6)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(0, 5)
    return g

def all_csets(g):
    return {str(v.index): [v.index] + g.neighbors(v) for v in g.vs}

def k_csets(g, k):
    vs = np.random.choice(g.vs, k, replace=False)
    return {str(v.index): [v.index] + g.neighbors(v) for v in vs}

def profile_aux():
    NSIM_NONAD = 100
    NITER = 10
    _, g = util.get_tc('B_A', 100, 2, directed=False)
    compare(g, all_csets(g), NSIM_NONAD, NITER)

def profile():
    prof.run('maxcut.profile_aux()', sort='cumulative')

def run(g, reps, niter, nsim_nonad, n_available, k):
    res = {'f_nonad': [], 'f_ad': [], 'imp': []}
    for rep in range(reps):
        print 'Rep:', rep
#        ig.plot(g)
#        xs, ys = zip(*[(left, count) for left, _, count in 
#                       g.degree_distribution().bins()])
#        plt.loglog(xs, ys, 'o')
#        plt.show()
        r = compare(g, k_csets(g, n_available), nsim_nonad, niter, k)
        res['f_nonad'].append(r['f_nonad'])
        res['f_ad'].append(r['f_ad'])
        res['imp'].append(r['imp'])
        print '===============================\n'
    return res

RESULT_DIR = os.path.abspath('../results/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Influence maximization')
    parser.add_argument('-m', '--model',
                        dest='model',
                        default='B_A',
                        help='Graph type')
    parser.add_argument('-n', '--nodes',
                        dest='nodes',
                        default=None,
                        type=int,
                        help='Number of nodes')
    parser.add_argument('-nn', '--nnonadaptive',
                        dest='nsim_nonad',
                        default=1000,
                        type=int,
                        help='Number of non-adaptive instances drawn')
    parser.add_argument('-ni', '--niter',
                        dest='niter',
                        default=100,
                        type=int,
                        help='Number of evaluation instances')
    parser.add_argument('-r', '--reps',
                        dest='reps',
                        default=10,
                        type=int,
                        help='Number of evaluation instances')
    parser.add_argument('-nav', '--navailable',
                        dest='navailable',
                        default=100,
                        type=int,
                        help='Number of available nodes to be cut')
    parser.add_argument('-k', '--k',
                        dest='k',
                        default=10,
                        type=int,
                        help='Cardinality constraint')
    args = parser.parse_args()
    REPS = args.reps
    NSIM_NONAD = args.nsim_nonad
    NITER = args.niter
    K = args.k
    N_AVAILABLE = args.navailable
    MODEL = args.model
    NODES = args.nodes

    (name, g) = util.get_tc(MODEL, NODES, directed=False)
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()

    res = run(g, REPS, NITER, NSIM_NONAD, N_AVAILABLE, K)
        
    now = time.localtime()
    fname = name + '_'
    fname += '{0:02d}{1}_'.format(now.tm_mday, cal.month_name[now.tm_mon][:3])
    fname += '{0:02d}_{1:02d}_{2:02d}'.format(now.tm_hour,
                                              now.tm_min,
                                              now.tm_sec)
    with open(os.path.join(RESULT_DIR, fname + '.pickle'), 'w') as f:
        pcl.dump(res, f)

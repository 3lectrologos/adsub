import argparse
import os
import sys
import random
import time
import cProfile as prof
import uuid
import subprocess as sub
import pickle

import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import joblib
import bitarray as ba
import progressbar

import submod
import util

DEBUG = False


def random_instance(g, p, active=None, copy=False, ret=False):
    if copy: g = g.copy()
    to_remove = []
    for e in g.es:
        if active == None or e.source not in active:
            if random.random() > p:
                to_remove.append(e.tuple)
    g.delete_edges(to_remove)
    if ret:
        return (g, to_remove)
    else:
        return g

def ic(g, a):
    r = list(set(a))
    for v in a:
        r.extend(g.subcomponent(v, mode=ig.OUT))
        # Just to avoid the list getting too long
        r = list(set(r))
    return set(r)

def ic_sim(g, p, niter, active=None, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    csim = np.zeros(shape=(n, n, niter), dtype='bool')
    for i in range(niter):
        (g, rem) = random_instance(g, p, active=active, copy=False, ret=True)
        for v in g.vs:
            # XXX: This assignment is the bottleneck here!
            csim[v.index, g.subcomponent(v, mode=ig.OUT), i] = True
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_sim_cond(g, p, niter, active, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    rest = [v.index for v in g.vs if v.index not in active]
    csim = {v: 0 for v in rest}
    for i in range(niter):
        (g, rem) = random_instance(g, p, active=active, copy=False, ret=True)
        for v in rest:
            # XXX: This assignment is the bottleneck here!
            csim[v] += len(set(g.subcomponent(v, mode=ig.OUT)) | active)
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    for v in csim:
        csim[v] /= (1.0*niter)
    if pbar:
        pb.finish()
    return csim

# Global weighting factor of node cost
LAMBDA = 1.1

def f_ic_base(h, a):
    active = ic(h, a)
    return len(active) - LAMBDA*len(a)

def f_ic(v, a, csim, prev):
    if prev == None:
        act = csim[v, :, :]
    else:
        act = prev | csim[v, :, :]
    return (1.0*np.sum(act))/csim.shape[2] - LAMBDA*(len(a) + 1)

def f_ic_ad(v, a, csim, active, fprev):
    a = set(a)
    active = list(set(active))
    if v in active:
        return fprev
    return csim[v] - LAMBDA*(len(a) + 1)

class BaseInfluence(submod.AdaptiveMax):
    pass

class NonAdaptiveInfluence(BaseInfluence):
    def __init__(self, g, p, nsim):
        super(NonAdaptiveInfluence, self).__init__([v.index for v in g.vs])
        self.g = g
        self.p = p
        self.nsim = nsim

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.csim = ic_sim(self.g, self.p, self.nsim, pbar=True)
        self.prev = None
        self.fsol = 0
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev)

    def update_f_hook(self):
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        if self.prev == None:
            self.prev = self.csim[self.sol[-1], :, :]
        else:
            self.prev = self.prev | self.csim[self.sol[-1], :, :]
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev)
        

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, nsim):
        super(AdaptiveInfluence, self).__init__([v.index for v in g.vs])
        self.g = g
        self.p = p
        self.nsim = nsim
        self.h = h

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic(self.h, self.sol)
        #print 'sol =', len(self.sol), ', active =', len(active)
        csim = ic_sim_cond(self.h, self.p, self.nsim, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol)
        self.fsol = len(active) - LAMBDA*len(self.sol)

def compare_worker(i, g, pedge, nsim_ad, v_nonad, k_ratio):
    print '-> worker', i, 'started.'
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, nsim_ad)
    (v_ad, _) = solver_ad.random_greedy(g.vcount()/k_ratio)
    active_nonad = ic(h, v_nonad)
    active_ad = ic(h, v_ad)
    if DEBUG:
        print 'Non-adaptive: vs  =', v_nonad
        print '              val =', f_ic_base(h, v_nonad)
        print 'Adaptive:     vs  =', v_ad
        print '              val =', f_ic_base(h, v_ad)
    if f_ic_base(h, v_ad) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': v_ad,
            'f_nonad': f_ic_base(h, v_nonad),
            'f_ad': f_ic_base(h, v_ad)}

def compare(g, pedge, nsim_nonad, nsim_ad, niter, k_ratio, parallel=True):
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.vs:
        st_nonad[v.index] = 0
        st_ad[v.index] = 0
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, pedge, nsim_nonad)
    (v_nonad, _) = solver_nonad.random_greedy(g.vcount()/k_ratio)
    # Adaptive simulation
    arg = [g, pedge, nsim_ad, v_nonad, k_ratio]
    if parallel:
        res = joblib.Parallel(n_jobs=4)((compare_worker, [i] + arg, {})
                                        for i in range(niter))
    else:
        res = [compare_worker(*([i] + arg)) for i in range(niter)]
    # Print results
    r_nonad = [r['f_nonad'] for r in res]
    r_ad = [r['f_ad'] for r in res]
    v_ad = [len(r['v_ad']) for r in res]
    return {
        'r_nonad': r_nonad,
        'v_nonad': v_nonad,
        'r_ad': r_ad,
        'v_ad': v_ad
        }

def profile_aux():
#    g = ig.Graph.Barabasi(50, 2)
    NODES = 100
    tmp = nx.barabasi_albert_graph(NODES, 2)
    g = ig.Graph(directed=True)
    g.add_vertices(NODES)
    for u, v in tmp.edges_iter():
        g.add_edge(u, v)
        g.add_edge(v, u)
    P_EDGE = 0.4
    NSIM_NONAD = 1000
    NSIM_AD = 1000
    NITER = 10
    K_RATIO = 5
    PARALLEL = False
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, PARALLEL)

def profile():
    prof.run('influence.profile_aux()', sort='cumulative')

def test_graph():
    name = 'TEST_GRAPH'
    g = ig.Graph()
    g.add_vertices(7)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(4, 1)
    g.add_edge(4, 2)
    g.add_edge(4, 3)
    g.add_edge(5, 6)
    g.to_directed()
    return (name, g)

def ba_graph(n, m):
    name = 'B_A_{0}_{1}'.format(n, m)
    g = ig.Graph.Barabasi(n, m)
    g.to_directed()
    return (name, g)

def format_result(r):
    sqn = np.sqrt(r['niter'])

    lon = 'Adaptive     |       favg = {0:.4g} '.format(np.mean(r['r_ad']))
    lon +=  '+/- {0:.3g}\n'.format(np.std(r['r_ad'])/sqn)
    bar = '-'*len(lon) + '\n'

    s = ''
    s += '{0} ({1})\n'.format(r['name'], r['git'])
    s += bar
    s += 'Nodes             : {0}\n'.format(r['vcount'])
    s += 'Edges             : {0}\n'.format(r['ecount'])
    s += 'Transitivity      : {0:.3g}\n'.format(r['trans'])
    s += 'Edge prob.        : {0:.3g}\n'.format(r['pedge'])
    s += 'Sims non-adaptive : {0}\n'.format(r['nsim_nonad'])
    s += 'Sims adaptive     : {0}\n'.format(r['nsim_ad'])
    s += 'Iterations        : {0}\n'.format(r['niter'])
    s += 'Lambda            : {0}\n'.format(r['lambda'])
    s += 'k-ratio           : {0}\n'.format(r['kratio'])
    s += 'Parallel          : {0}\n'.format(r['parallel'])
    s += 'Time taken        : {0}h{1}m{2}s\n'.format(r['t_h'],
                                                     r['t_m'],
                                                     r['t_s'])
    s += bar
    s += 'Non-adaptive |       favg = {0:.4g} '.format(np.mean(r['r_nonad']))
    s +=  '+/- {0:.3g}\n'.format(np.std(r['r_nonad'])/sqn)
    s += '             |     #nodes = {0:.3g}\n'.format(len(r['v_nonad']))
    s += bar
    s += lon
    s += '             | avg #nodes = {0:.3g}\n'.format(np.mean(r['v_ad']))
    s += bar
    return s

RESULT_DIR = os.path.abspath('../results/')

# TODO: Adjust drawn samples in adaptive case according to "remaining"
#       nodes or edges
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Influence maximization')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug')
    args = parser.parse_args()
    DEBUG = args.debug

    random.seed(0)
    np.random.seed(0)
    P_EDGE = 0.25
    NSIM_NONAD = 200
    NSIM_AD = 100
    NITER = 10
    K_RATIO = 5
    PARALLEL = True

    (name, g) = ba_graph(100, 2)
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()

    stime = time.time()
    r = compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, PARALLEL)
    etime = time.time()
    dt = int(etime - stime)
    hours = dt / 3600
    mins = (dt % 3600) / 60
    secs = (dt % 3600) % 60

    r['name'] = name
    r['vcount'] = g.vcount()
    r['ecount'] = g.ecount()
    r['trans'] = g.transitivity_undirected()
    r['pedge'] = P_EDGE
    r['nsim_nonad'] = NSIM_NONAD
    r['nsim_ad'] = NSIM_AD
    r['niter'] = NITER
    r['lambda'] = LAMBDA
    r['kratio'] = K_RATIO
    r['parallel'] = PARALLEL
    r['t_h'] = hours
    r['t_m'] = mins
    r['t_s'] = secs
    r['git'] = sub.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    
    s = format_result(r)
    print s
    fname = name + '_' + str(uuid.uuid1())
    with open(os.path.join(RESULT_DIR, fname + '.txt'), 'w') as f:
        f.write(s)
    with open(os.path.join(RESULT_DIR, fname + '.pickle'), 'w') as f:
        pickle.dump(r, f)

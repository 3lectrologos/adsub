import argparse
import os
import sys
import random
import time
import cProfile as prof
import uuid
import subprocess as sub
import pickle
import calendar as cal

import networkx as nx
import igraph as ig
import numpy as np
import joblib
import bitarray as ba
import progressbar

import submod
import util

DEBUG = False


def random_instance(g, p, copy=False, ret=False):
    if copy: g = g.copy()
    to_remove = []
    for e in g.es:
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

def ic_sim(g, p, niter, pbar=False):
    if pbar:
        pb = progressbar.ProgressBar(maxval=niter).start()
    n = g.vcount()
    csim = np.zeros(shape=(n, n, niter), dtype='bool')
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in g.vs:
            # XXX: This assignment is the bottleneck here!
            csim[v['i'], g.subcomponent(v, mode=ig.OUT), i] = True
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_sim_cond(g, p, niter, active):
    n = g.vcount()
    rest = [v for v in g.vs if v['i'] not in active]
    csim = {v['i']: 0 for v in rest}
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in rest:
            # XXX: This assignment is the bottleneck here!
            a = set(g.vs[u]['i'] for u in g.subcomponent(v.index, mode=ig.OUT))
            csim[v['i']] += len(a | active)
        g.add_edges(rem)
    for v in csim:
        csim[v] /= (1.0*niter)
    return csim

def delete_active(g, active):
    g.delete_vertices([v for v in g.vs if v['i'] in active])

def f_comb(val, cost, GAMMA):
    return val - GAMMA*cost

def f_ic_base(h, a, fc):
    active = ic(h, a)
    return fc(len(active), len(a))

def f_ic(v, a, csim, prev, fc):
    if prev == None:
        act = csim[v, :, :]
    else:
        act = prev | csim[v, :, :]
    return fc((1.0*np.sum(act))/csim.shape[2], len(a) + 1)

def f_ic_ad(v, a, csim, active, fprev, fc):
    a = set(a)
    active = list(set(active))
    if v in active:
        return fprev
    return fc(csim[v], len(a) + 1)

class BaseInfluence(submod.AdaptiveMax):
    pass

class NonAdaptiveInfluence(BaseInfluence):
    def __init__(self, g, p, fc, nsim):
        super(NonAdaptiveInfluence, self).__init__([v['i'] for v in g.vs])
        self.g = g
        self.p = p
        self.fc = fc
        self.nsim = nsim

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.csim = ic_sim(self.g, self.p, self.nsim, pbar=True)
        self.prev = None
        self.fsol = 0
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev, self.fc)

    def update_f_hook(self):
        self.fsol = self.f(self.sol[-1], self.sol[:-1])
        if self.prev == None:
            self.prev = self.csim[self.sol[-1], :, :]
        else:
            self.prev = self.prev | self.csim[self.sol[-1], :, :]
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev, self.fc)
        

class AdaptiveInfluence(BaseInfluence):
    def __init__(self, g, h, p, fc, nsim):
        super(AdaptiveInfluence, self).__init__([v['i'] for v in g.vs])
        self.g = g.copy()
        self.h = h
        self.p = p
        self.fc = fc
        self.nsim = nsim

    def init_f_hook(self):
        super(AdaptiveInfluence, self).init_f_hook()
        self.update_f_hook()

    def update_f_hook(self):
        active = ic(self.h, self.sol)
        print '#sol =', len(self.sol), ', #active =', len(active)
        delete_active(self.g, active)
        csim = ic_sim_cond(self.g, self.p, self.nsim, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol, self.fc)
        self.fsol = self.fc(len(active), len(self.sol))
        print 'fsol =', self.fsol

def compare_worker(i, g, pedge, nsim_ad, v_nonad, k_ratio, fc):
    print '-> worker', i, 'started.'
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, fc, nsim_ad)
    (v_ad, _) = solver_ad.random_greedy(g.vcount()/k_ratio)
    active_nonad = ic(h, v_nonad)
    active_ad = ic(h, v_ad)
    if DEBUG:
        print 'Non-adaptive: vs  =', v_nonad
        print '              val =', f_ic_base(h, v_nonad, fc)
        print 'Adaptive:     vs  =', v_ad
        print '              val =', f_ic_base(h, v_ad, fc)
    if f_ic_base(h, v_ad, fc) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    return {'active_nonad': active_nonad,
            'active_ad': active_ad,
            'v_ad': v_ad,
            'f_nonad': f_ic_base(h, v_nonad, fc),
            'f_ad': f_ic_base(h, v_ad, fc)}

def compare(g, pedge, nsim_nonad, nsim_ad, niter, k_ratio, gamma, parallel=True):
    f_nonad = []
    f_ad = []
    v_ad = []
    st_nonad = {}
    st_ad = {}
    for v in g.vs:
        st_nonad[v['i']] = 0
        st_ad[v['i']] = 0
    fc = lambda a, b: f_comb(a, b, gamma)
    # Non-adaptive simulation
    solver_nonad = NonAdaptiveInfluence(g, pedge, fc,  nsim_nonad)
    (v_nonad, _) = solver_nonad.random_greedy(g.vcount()/k_ratio)
    # Adaptive simulation
    arg = [g, pedge, nsim_ad, v_nonad, k_ratio, fc]
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
    random.seed(0)
    np.random.seed(0)
    P_EDGE = 0.3
    NSIM_NONAD = 1000
    NSIM_AD = 1000
    NITER = 10
    K_RATIO = 10
    GAMMA = 1
    PARALLEL = False
    (name, g) = tc_ba(100, 2)
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, GAMMA, PARALLEL)

def profile():
    prof.run('influence.profile_aux()', sort='cumulative')

def test_graph():
    g = ig.Graph()
    g.add_vertices(7)
    for v in g.vs:
        v['i'] = v.index
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(4, 1)
    g.add_edge(4, 2)
    g.add_edge(4, 3)
    g.add_edge(5, 6)
    g.to_directed()
    return g

def tc_test():
    name = 'TEST_GRAPH'
    return (name, test_graph())

def tc_ba(n, m):
    name = 'B_A_{0}_{1}'.format(n, m)
    g = ig.Graph.Barabasi(n, m)
    g.to_directed()
    for v in g.vs:
        v['i'] = v.index
    return (name, g)

def tc_snap_gr(n):
    name = 'SNAP_GR_' + str(n)
    fpath = os.path.join(DATA_DIR, 'general_relativity.txt')
    g = util.read_snap_graph(fpath)
    rem = [v for v in g.vs[n:]]
    g.delete_vertices(rem)
    for v in g.vs:
        v['i'] = v.index
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
    s += 'Gamma             : {0}\n'.format(r['gamma'])
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

DATA_DIR = os.path.abspath('../data/')
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
    P_EDGE = 0.05
    NSIM_NONAD = 10000
    NSIM_AD = 1000
    NITER = 100
    K_RATIO = 10
    GAMMA = 3
    PARALLEL = True

    (name, g) = tc_snap_gr(10)
#    (name, g) = ba_graph(100, 2)
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()

    stime = time.time()
    r = compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, GAMMA, PARALLEL)
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
    r['gamma'] = GAMMA
    r['kratio'] = K_RATIO
    r['parallel'] = PARALLEL
    r['t_h'] = hours
    r['t_m'] = mins
    r['t_s'] = secs
    r['git'] = sub.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    
    s = format_result(r)
    print s

    now = time.localtime()
    fname = name + '_'
    fname += '{0:02d}{1}_'.format(now.tm_mday, cal.month_name[now.tm_mon][:3])
    fname += '{0}_{1}_{2}'.format(now.tm_hour, now.tm_min, now.tm_sec)
    with open(os.path.join(RESULT_DIR, fname + '.txt'), 'w') as f:
        f.write(s)
    with open(os.path.join(RESULT_DIR, fname + '.pickle'), 'w') as f:
        pickle.dump(r, f)

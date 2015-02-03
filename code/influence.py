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
    csim = {v['i']: {} for v in g.vs}
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in g.vs:
            b = n*ba.bitarray('0')
            for z in g.subcomponent(v, mode=ig.OUT):
                b[z] = True
            csim[v['i']][i] = b
        g.add_edges(rem)
        if pbar:
            pb.update(i)
    if pbar:
        pb.finish()
    return csim

def ic_sim_cond(g, p, niter, active):
    n = g.vcount()
    csim = {v.index: 0 for v in g.vs}
    for i in range(niter):
        (g, rem) = random_instance(g, p, copy=False, ret=True)
        for v in csim:
            csim[v] += len(g.subcomponent(v, mode=ig.OUT))
        g.add_edges(rem)
    d = {}
    for v in g.vs:
        d[v['i']] = len(active) + csim[v.index]/(1.0*niter)
    return d

def delete_active(g, active):
    g.delete_vertices([v for v in g.vs if v['i'] in active])

def f_comb(val, cost, GAMMA):
    return val - GAMMA*cost

def f_ic_base(h, a, fc):
    active = ic(h, a)
    return fc(len(active), len(a))

def f_ic(v, a, csim, prev, fc, ret=False):
    s = 0
    vcount = len(csim)
    nsim = len(csim[v])
    if prev != None:
        for k, b in csim[v].iteritems():
            if ret:
                prev[k] |= b
                s += prev[k].count(1)
            else:
                s += (prev[k] | b).count(1)
    else:
        if ret:
            prev = {i: vcount*ba.bitarray('0') for i in range(nsim)}
        for k, b, in csim[v].iteritems():
            s += b.count(1)
            if ret:
                prev[k] |= b
    if ret:
        return (fc((1.0*s)/nsim, len(a) + 1), prev)
    else:
        return fc((1.0*s)/nsim, len(a) + 1)

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
        self.csim = ic_sim(self.g, self.p, self.nsim, pbar=True)

    def init_f_hook(self):
        super(NonAdaptiveInfluence, self).init_f_hook()
        self.prev = None
        self.f = lambda v, a: f_ic(v, a, self.csim, self.prev, self.fc)

    def update_f_hook(self):
        (self.fsol, self.prev) = f_ic(self.sol[-1], self.sol[:-1], self.csim,
                                      self.prev, self.fc, ret=True)
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
#        print '#sol =', len(self.sol), ', #active =', len(active)
        delete_active(self.g, active)
        csim = ic_sim_cond(self.g, self.p, self.nsim, active=active)
        self.f = lambda v, a: f_ic_ad(v, a, csim, active, self.fsol, self.fc)
        self.fsol = self.fc(len(active), len(self.sol))
#        print 'fsol =', self.fsol

def compare_worker(i, g, pedge, nsim_ad, v_nonad_rg, v_nonad_g, v_nonad_r, k_ratio, gamma):
    print '-> worker', i, 'started.'
    fc = lambda a, b: f_comb(a, b, gamma)
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, fc, nsim_ad)
    (v_ad, _) = solver_ad.random_greedy(g.vcount()/k_ratio)
    active_nonad_rg = ic(h, v_nonad_rg)
    active_nonad_g = ic(h, v_nonad_g)
    active_nonad_r = ic(h, v_nonad_r)
    active_ad = ic(h, v_ad)
    if f_ic_base(h, v_ad, fc) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    return {
        'active_nonad_rg': active_nonad_rg,
        'active_nonad_g': active_nonad_g,
        'active_nonad_r': active_nonad_r,
        'active_ad': active_ad,
        'v_ad': v_ad,
        'f_nonad_rg': f_ic_base(h, v_nonad_rg, fc),
        'f_nonad_g': f_ic_base(h, v_nonad_g, fc),
        'f_nonad_r': f_ic_base(h, v_nonad_r, fc),
        'f_ad': f_ic_base(h, v_ad, fc)
        }

def compare(g, pedge, nsim_nonad, nsim_ad, niter, k_ratio, gamma, workers=0):
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
    (v_nonad_rg, _) = solver_nonad.random_greedy(g.vcount()/k_ratio)
    (v_nonad_g, _) = solver_nonad.greedy(g.vcount()/k_ratio)
    (v_nonad_r, _) = solver_nonad.random(g.vcount()/k_ratio)
    del solver_nonad.csim
    # Adaptive simulation
    arg = [g, pedge, nsim_ad, v_nonad_rg, v_nonad_g, v_nonad_r, k_ratio, gamma]
    if workers > 1:
        res = joblib.Parallel(n_jobs=workers)((compare_worker, [i] + arg, {})
                                        for i in range(niter))
    else:
        res = [compare_worker(*([i] + arg)) for i in range(niter)]
    # Return results
    r_nonad_rg = [r['f_nonad_rg'] for r in res]
    r_nonad_g = [r['f_nonad_g'] for r in res]
    r_nonad_r = [r['f_nonad_r'] for r in res]
    r_ad = [r['f_ad'] for r in res]
    v_ad = [len(r['v_ad']) for r in res]
    return {
        'r_nonad_rg': r_nonad_rg,
        'v_nonad_rg': v_nonad_rg,
        'r_nonad_g': r_nonad_g,
        'v_nonad_g': v_nonad_g,
        'r_nonad_r': r_nonad_r,
        'v_nonad_r': v_nonad_r,
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
    GAMMA = 3
    (name, g) = tc_snap_gr(100)
    compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, GAMMA)

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

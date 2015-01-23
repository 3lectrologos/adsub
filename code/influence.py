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

def compare_worker(i, g, pedge, nsim_ad, v_nonad_rg, v_nonad_g, k_ratio, gamma):
    print '-> worker', i, 'started.'
    fc = lambda a, b: f_comb(a, b, gamma)
    h = random_instance(g, pedge, copy=True)
    solver_ad = AdaptiveInfluence(g, h, pedge, fc, nsim_ad)
    (v_ad, _) = solver_ad.random_greedy(g.vcount()/k_ratio)
    active_nonad_rg = ic(h, v_nonad_rg)
    active_nonad_g = ic(h, v_nonad_g)
    active_ad = ic(h, v_ad)
    if f_ic_base(h, v_ad, fc) != solver_ad.fsol:
        raise 'Inconsistent adaptive function values'
    return {
        'active_nonad_rg': active_nonad_rg,
        'active_nonad_g': active_nonad_g,
        'active_ad': active_ad,
        'v_ad': v_ad,
        'f_nonad_rg': f_ic_base(h, v_nonad_rg, fc),
        'f_nonad_g': f_ic_base(h, v_nonad_g, fc),
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
    del solver_nonad.csim
    # Adaptive simulation
    arg = [g, pedge, nsim_ad, v_nonad_rg, v_nonad_g, k_ratio, gamma]
    if workers > 1:
        res = joblib.Parallel(n_jobs=workers)((compare_worker, [i] + arg, {})
                                        for i in range(niter))
    else:
        res = [compare_worker(*([i] + arg)) for i in range(niter)]
    # Return results
    r_nonad_rg = [r['f_nonad_rg'] for r in res]
    r_nonad_g = [r['f_nonad_g'] for r in res]
    r_ad = [r['f_ad'] for r in res]
    v_ad = [len(r['v_ad']) for r in res]
    return {
        'r_nonad_rg': r_nonad_rg,
        'r_nonad_g': r_nonad_g,
        'v_nonad_rg': v_nonad_rg,
        'v_nonad_g': v_nonad_g,
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
    s += 'Workers           : {0}\n'.format(r['workers'])
    s += 'Time taken        : {0}h{1}m{2}s\n'.format(r['t_h'],
                                                     r['t_m'],
                                                     r['t_s'])
    s += bar
    s += 'Non-adaptive |       favg = {0:.4g} '.format(np.mean(r['r_nonad_g']))
    s +=  '+/- {0:.3g}\n'.format(np.std(r['r_nonad_g'])/sqn)
    s += '     (g)     |     #nodes = {0:.3g}\n'.format(len(r['v_nonad_g']))
    s += bar
    s += bar
    s += 'Non-adaptive |       favg = {0:.4g} '.format(np.mean(r['r_nonad_rg']))
    s +=  '+/- {0:.3g}\n'.format(np.std(r['r_nonad_rg'])/sqn)
    s += '    (rg)     |     #nodes = {0:.3g}\n'.format(len(r['v_nonad_rg']))
    s += bar

    s += lon
    s += '             | avg #nodes = {0:.3g}\n'.format(np.mean(r['v_ad']))
    s += bar
    return s

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
    parser.add_argument('-p', '--pedge',
                        dest='p_edge',
                        default=0.05,
                        type=float,
                        help='Probability of each edge being removed')
    parser.add_argument('-nn', '--nnonadaptive',
                        dest='nsim_nonad',
                        default=1000,
                        type=int,
                        help='Number of non-adaptive instances drawn')
    parser.add_argument('-na', '--nadaptive',
                        dest='nsim_ad',
                        default=1000,
                        type=int,
                        help='Number of instances drawn at each adaptive step')
    parser.add_argument('-ni', '--niter',
                        dest='niter',
                        default=8,
                        type=int,
                        help='Number of evaluation instances')
    parser.add_argument('-k', '--kratio',
                        dest='k_ratio',
                        default=10,
                        type=int,
                        help='Cardinality constraint as a % of #nodes')
    parser.add_argument('-g', '--gamma',
                        dest='gamma',
                        default=2,
                        type=float,
                        help='Cost per node')
    parser.add_argument('-w', '--workers',
                        dest='workers',
                        default=1,
                        type=int,
                        help='Number of parallel workers (1 = sequential)')
    args = parser.parse_args()
    P_EDGE = args.p_edge
    NSIM_NONAD = args.nsim_nonad
    NSIM_AD = args.nsim_ad
    NITER = args.niter
    K_RATIO = args.k_ratio
    GAMMA = args.gamma
    WORKERS = args.workers
    MODEL = args.model
    NODES = args.nodes

    random.seed(0)
    np.random.seed(0)

    (name, g) = util.get_tc(MODEL, NODES)

    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()

    stime = time.time()
    r = compare(g, P_EDGE, NSIM_NONAD, NSIM_AD, NITER, K_RATIO, GAMMA, WORKERS)
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
    r['workers'] = WORKERS
    r['t_h'] = hours
    r['t_m'] = mins
    r['t_s'] = secs
    r['git'] = sub.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    
    s = format_result(r)
    print s

    now = time.localtime()
    fname = name + '_'
    fname += '{0:02d}{1}_'.format(now.tm_mday, cal.month_name[now.tm_mon][:3])
    fname += '{0:02d}_{1:02d}_{2:02d}'.format(now.tm_hour,
                                              now.tm_min,
                                              now.tm_sec)
    with open(os.path.join(RESULT_DIR, fname + '.txt'), 'w') as f:
        f.write(s)
    with open(os.path.join(RESULT_DIR, fname + '.pickle'), 'w') as f:
        pickle.dump(r, f)

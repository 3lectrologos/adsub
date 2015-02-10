#!/usr/bin/env python
import argparse
import os
import re
import cPickle as pcl
import numpy as np
import igraph as ig
import util
import influence
import maxcut


TEMPLATE_FS = 'fs.tex'


INF_FAST = dict(reps=5,
                pedge=0.3,
                gamma=1,
                nsim_nonad=100,
                nsim_ad=50,
                niter=20,
                workers=7)
INF_SLOW = dict(reps=20,
                pedge=0.15,
                gamma=1,
                nsim_nonad=1000,
                nsim_ad=100,
                niter=100,
                workers=7)
MC_FAST = dict(reps=10,
               nsim_nonad=100,
               niter=50)
MC_SLOW = dict(reps=30,
               nsim_nonad=1000,
               niter=100)


def print_info(name, g):
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()
    print 'assortativity =', g.assortativity_degree()


def run_inf(model, nodes, fast=True, plot=False):
    (name, g) = util.get_tc(model, nodes, directed=False)
    if plot:
        ig.plot(g)
        util.plot_degree_dist(g)
    outdir = os.path.join(util.DIR_RES, 'influence', model)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    g.write_edgelist(os.path.join(outdir, 'graph'))
    xs = []
    fs_rand = []
    fs_nonad = []
    fs_ad = []
    n_available = 50#len(g.vs) / 10
    pcts = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
    ks = list(np.unique([max(1, int(kr * n_available)) for kr in pcts]))
    print_info(name, g)
    for k in ks:
        if fast:
            params = INF_FAST
        else:
            params = INF_SLOW
        r = influence.run(g, n_available=n_available, k=k, **params)
        fs_rand += r['f_rand']
        fs_nonad += r['f_nonad']
        fs_ad += r['f_ad']
        xs += [k] * params['reps']
    data = {
        'model': model,
        'ks': xs,
        'rand': fs_rand,
        'nonad': fs_nonad,
        'ad': fs_ad
        }
    plot_fs(data, outdir)
    with open(os.path.join(outdir, model.lower() + '.pickle'), 'w') as f:
        pcl.dump(data, f)


def run_mc(model, nodes, fast=True, plot=False):
    (name, g) = util.get_tc(model, nodes, directed=False)
    if plot:
        ig.plot(g)
        util.plot_degree_dist(g)
    outdir = os.path.join(util.DIR_RES, 'maxcut', model)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    g.write_edgelist(os.path.join(outdir, 'graph'))
    xs = []
    fs_rand = []
    fs_nonad = []
    fs_ad = []
    n_available = 50#len(g.vs) / 10
    pcts = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
    ks = list(np.unique([max(1, int(kr * n_available)) for kr in pcts]))
    print_info(name, g)
    for k in ks:
        if fast:
            params = MC_FAST
        else:
            params = MC_SLOW
        r = maxcut.run(g, n_available=n_available, k=k, **params)
        fs_rand += r['f_rand']
        fs_nonad += r['f_nonad']
        fs_ad += r['f_ad']
        xs += [k] * params['reps']
    data = {
        'model': model,
        'ks': xs,
        'rand': fs_rand,
        'nonad': fs_nonad,
        'ad': fs_ad
        }
    plot_fs(data, outdir)
    with open(os.path.join(outdir, model.lower() + '.pickle'), 'w') as f:
        pcl.dump(data, f)


def plot_fs(data, outdir):
    cs = {}
    for name in ['rand', 'nonad', 'ad']:
        cs[name] = util.get_coords(data['ks'], data[name])
        means = []
        ks = np.unique(data['ks'])
        for k in ks:
            means.append(np.mean(np.array(data[name])[data['ks'] == k]))
        cs['mean_' + name] = util.get_coords(ks, means)
    cs['x_max'] = str(max(data['ks']))
    cs['y_max'] = str(max(data['ad']))
    cs['title'] = re.escape(data['model'])
    texname = 'fs_' + data['model'].lower() + '.tex'
    util.replace(outdir, TEMPLATE_FS, cs, texname)
    util.maketex(outdir, texname)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Adaptive monotone submodular experiments')
    parser.add_argument('objective',
                        choices=['inf', 'mc'],
                        help='objective function to maximize')
    parser.add_argument('model',
                        help='data set')
    parser.add_argument('nodes',
                        type=int,
                        help='number of nodes to subsample')
    parser.add_argument('-s', '--slow',
                        action='store_true',
                        help='more detailed simulation')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='plot graph and degree distribution')
    args = parser.parse_args()
    objective = args.objective
    model = args.model
    nodes = args.nodes
    plot = args.plot
    fast = not args.slow
    runargs = [model, nodes, fast, plot]
    if objective == 'inf':
        run_inf(*runargs)
    elif objective == 'mc':
        run_mc(*runargs)

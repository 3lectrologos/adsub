#!/usr/bin/env python
import argparse
import os
import re
import numpy as np
import igraph as ig
import util
import influence
import maxcut


TEMPLATE_INF_IMP = 'inf_imp.tex'
TEMPLATE_INF_FS = 'inf_fs.tex'
TEMPLATE_INF_VS = 'inf_vs.tex'
TEMPLATE_MC_IMP = 'mc_imp.tex'
TEMPLATE_MC_FS = 'mc_fs.tex'

INF_FAST = dict(nsim_nonad=100,
                nsim_ad=10,
                niter=20,
                k_ratio=20,
                gamma=1,
                workers=7)
INF_SLOW = dict(nsim_nonad=1000,
                nsim_ad=100,
                niter=50,
                k_ratio=20,
                gamma=1,
                workers=7)
MC_FAST = dict(reps=3,
               nsim_nonad=10,
               niter=20)
MC_SLOW = dict(reps=20,
               nsim_nonad=100,
               niter=100)


def run_inf(model, nodes, fast=True, plot=False):
    (name, g) = util.get_tc(model, nodes, directed=False)
    if plot:
        ig.plot(g)
        util.plot_degree_dist(g)

    outdir = os.path.join(util.DIR_RES, 'influence', model)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    g.write_edgelist(os.path.join(outdir, 'graph'))

    imps = []
    fs_nonad = []
    fs_ad = []
    fs_rand = []
    vs_nonad = []
    vs_ad = []
    ps = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()
    for p in ps:
        if fast:
            params = INF_FAST
        else:
            params = INF_SLOW
        r = influence.compare(g, p, **params)
        fm_nonad = np.mean(r['r_nonad_rg'])
        fm_ad = np.mean(r['r_ad'])
        fm_rand = np.mean(r['r_nonad_r'])
        fs_nonad.append(fm_nonad)
        fs_ad.append(fm_ad)
        fs_rand.append(fm_rand)
        imp = 100.0 * (fm_ad - fm_nonad) / nodes
        imps.append(imp)
        vm_nonad = len(r['v_nonad_rg'])
        vs_nonad.append((1.0 * fm_nonad) / vm_nonad)
        fv_ratios = [(1.0 * x) / y for x, y in zip(r['r_ad'], r['v_ad'])]
        vs_ad.append(np.mean(fv_ratios))
        print 'p =', p
        print 'f (rand) =', fm_rand
        print 'f (nonad) =', fm_nonad
        print 'f (ad) =', fm_ad
        print 'imp =', imp
        print 'v (nonad) =', vm_nonad
        print 'v (ad) =', np.mean(r['v_ad'])
    esc_model = re.escape(model)
    # Improvements plot
    imps_plot = util.get_coords(ps, imps)
    texname = 'imps_' + model.lower() + '.tex'
    util.replace(outdir, TEMPLATE_INF_IMP, {'means': imps_plot,
                                            'title': esc_model},
                 outname=texname)
    util.maketex(outdir, texname)
    # f_avg plot
    fs_nonad_plot = util.get_coords(ps, fs_nonad)
    fs_ad_plot = util.get_coords(ps, fs_ad)
    fs_rand_plot = util.get_coords(ps, fs_rand)
    texname = 'fs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_INF_FS,
                 {'fs_nonad': fs_nonad_plot, 'fs_ad': fs_ad_plot,
                  'fs_rand': fs_rand_plot, 'title': esc_model},
                 outname=texname)
    util.maketex(outdir, texname)
    # f_avg / num. of nodes plot
    vs_nonad_plot = util.get_coords(ps, vs_nonad)
    vs_ad_plot = util.get_coords(ps, vs_ad)
    texname = 'vs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_INF_VS,
                 {'vs_nonad': vs_nonad_plot, 'vs_ad': vs_ad_plot,
                  'title': esc_model},
                 outname=texname)
    util.maketex(outdir, texname)


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
    imps = []
    imp_means = []
    fs_rand = []
    fs_rand_means = []
    fs_nonad = []
    fs_nonad_means = []
    fs_ad = []
    fs_ad_means = []
    n_available = len(g.vs) / 10
    ks = [max(1, int(kr * n_available)) for kr in [0.01, 0.1, 0.3, 0.5, 0.7, 1]]
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()
    for k in ks:
        if fast:
            params = MC_FAST
        else:
            params = MC_SLOW
        r = maxcut.run(g, n_available=n_available, k=k, **params)
        imps += r['imp']
        imp_means.append(np.mean(r['imp']))
        fs_rand += r['f_rand']
        fs_rand_means.append(np.mean(r['f_rand']))
        fs_nonad += r['f_nonad']
        fs_nonad_means.append(np.mean(r['f_nonad']))
        fs_ad += r['f_ad']
        fs_ad_means.append(np.mean(r['f_ad']))
        xs += [k] * params['reps']
    esc_model = re.escape(model)
    # Improvements plot
    imp_ms = util.get_coords(ks, imp_means)
    imp_cs = util.get_coords(xs, imps)
    texname = 'imp_' + model.lower() + '.tex'
    util.replace(outdir, TEMPLATE_MC_IMP, {'imps': imp_cs, 'imp_means': imp_ms,
                                           'title': esc_model},
                 outname=texname)
    util.maketex(outdir, texname)
    # f_avg plot
    fs_rand_ms = util.get_coords(ks, fs_rand_means)
    fs_rand_cs = util.get_coords(xs, fs_rand)
    fs_nonad_ms = util.get_coords(ks, fs_nonad_means)
    fs_nonad_cs = util.get_coords(xs, fs_nonad)
    fs_ad_ms = util.get_coords(ks, fs_ad_means)
    fs_ad_cs = util.get_coords(xs, fs_ad)
    texname = 'fs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_MC_FS,
                 {'fs_nonad': fs_nonad_cs, 'f_nonad_means': fs_nonad_ms,
                  'fs_rand': fs_rand_cs, 'f_rand_means': fs_rand_ms,
                  'fs_ad': fs_ad_cs, 'f_ad_means': fs_ad_ms,
                  'y_max': str(max(fs_ad)), 'title': esc_model},
                 outname=texname)
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

import argparse
import os
import numpy as np
import igraph as ig
import util
import maxcut

TEMPLATE_MC_IMP = 'mc_imp.tex'
TEMPLATE_MC_FS = 'mc_fs.tex'


def run(model, nodes):
    (name, g) = util.get_tc(model, nodes, directed=False)
    #    ig.plot(g)
    #    util.plot_degree_dist(g)

    outdir = os.path.join(util.DIR_RES, 'maxcut', model)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        g.write_edgelist(os.path.join(outdir, 'graph'))

    xs = []
    imps = []
    imp_means = []
    fs_nonad = []
    fs_nonad_means = []
    fs_ad = []
    fs_ad_means = []
    n_available = len(g.vs) / 10
    ks = [int(kr * n_available) for kr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]]
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()
    for k in ks:
        reps = 5
        niter = 10
        nsim_nonad = 10
        r = maxcut.run(g, reps, niter, nsim_nonad, n_available, k)
        imps += r['imp']
        imp_means.append(np.mean(r['imp']))
        fs_nonad += r['f_nonad']
        fs_nonad_means.append(np.mean(r['f_nonad']))
        fs_ad += r['f_ad']
        fs_ad_means.append(np.mean(r['f_ad']))
        xs += [k] * reps

    imp_ms = util.get_coords(ks, imp_means)
    imp_cs = util.get_coords(xs, imps)
    texname = 'imp_' + model.lower() + '.tex'
    util.replace(outdir, TEMPLATE_MC_IMP, {'imps': imp_cs, 'imp_means': imp_ms},
                 outname=texname)
    util.maketex(outdir, texname)

    fs_nonad_ms = util.get_coords(ks, fs_nonad_means)
    fs_nonad_cs = util.get_coords(xs, fs_nonad)
    fs_ad_ms = util.get_coords(ks, fs_ad_means)
    fs_ad_cs = util.get_coords(xs, fs_ad)
    texname = 'fs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_MC_FS,
                 {'fs_nonad': fs_nonad_cs, 'f_nonad_means': fs_nonad_ms,
                  'fs_ad': fs_ad_cs, 'f_ad_means': fs_ad_ms,
                  'y_max': str(max(fs_ad))},
                 outname=texname)
    util.maketex(outdir, texname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Max cut')
    parser.add_argument('-m', '--model',
                        help='Data set')
    parser.add_argument('-n', '--nodes',
                        dest='nodes',
                        type=int,
                        help='Number of nodes to subsample')
    args = parser.parse_args()
    model = args.model
    nodes = args.nodes
    run(model, nodes)

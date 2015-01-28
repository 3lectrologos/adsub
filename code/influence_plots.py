import argparse
import os
import numpy as np
import igraph as ig
import util
import influence


TEMPLATE_INF_IMP = 'inf_imp.tex'
TEMPLATE_INF_FS = 'inf_fs.tex'
TEMPLATE_INF_VS = 'inf_vs.tex'


def run(model, nodes):
    (name, g) = util.get_tc(model, nodes, directed=False)
    #ig.plot(g)
    #util.plot_degree_dist(g)

    outdir = os.path.join(util.DIR_RES, 'influence', model)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    g.write_edgelist(os.path.join(outdir, 'graph'))

    imps = []
    fs_nonad = []
    fs_ad = []
    vs_nonad = []
    vs_ad = []
    ps = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
    print 'Running {0}'.format(name)
    print '#nodes =', g.vcount()
    print '#edges =', g.ecount()
    print 'transitivity =', g.transitivity_undirected()
    for p in ps:
        niter = 20
        nsim_nonad = 100
        nsim_ad = 10
        k_ratio = 20
        gamma = 1
        n_workers = 4
        r = influence.compare(g, p, nsim_nonad, nsim_ad, niter, k_ratio, gamma,
                              n_workers)
        fm_nonad = np.mean(r['r_nonad_rg'])
        fm_ad = np.mean(r['r_ad'])
        fs_nonad.append(fm_nonad)
        fs_ad.append(fm_ad)
        imp = 100.0 * (fm_ad - fm_nonad) / nodes
        imps.append(imp)
        vm_nonad = len(r['v_nonad_rg'])
        vs_nonad.append((1.0 * fm_nonad) / vm_nonad)
        fv_ratios = [(1.0 * x) / y for x, y in zip(r['r_ad'], r['v_ad'])]
        vs_ad.append(np.mean(fv_ratios))
        print 'p =', p
        print 'f (nonad) =', fm_nonad
        print 'f (ad) =', fm_ad
        print 'imp =', imp
        print 'v (nonad) =', vm_nonad
        print 'v (ad) =', np.mean(r['v_ad'])

    imps_plot = util.get_coords(ps, imps)
    texname = 'imps_' + model.lower() + '.tex'
    util.replace(outdir, TEMPLATE_INF_IMP, {'means': imps_plot},
                 outname=texname)
    util.maketex(outdir, texname)

    fs_nonad_plot = util.get_coords(ps, fs_nonad)
    fs_ad_plot = util.get_coords(ps, fs_ad)
    texname = 'fs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_INF_FS,
                 {'fs_nonad': fs_nonad_plot, 'fs_ad': fs_ad_plot},
                 outname=texname)
    util.maketex(outdir, texname)

    vs_nonad_plot = util.get_coords(ps, vs_nonad)
    vs_ad_plot = util.get_coords(ps, vs_ad)
    texname = 'vs_' + model.lower() + '.tex'
    util.replace(outdir,
                 TEMPLATE_INF_VS,
                 {'vs_nonad': vs_nonad_plot, 'vs_ad': vs_ad_plot},
                 outname=texname)
    util.maketex(outdir, texname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Influence maximization')
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

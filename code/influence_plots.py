import os
import numpy as np
import igraph as ig
import util
import influence


FN_MC_IMP = 'inf_imp.tex'

model = 'EGO_FB'
nodes = 1000
(name, g) = util.get_tc(model, nodes, directed=False)
ig.plot(g)
util.plot_degree_dist(g)

outdir = os.path.join(util.DIR_RES, 'influence', model)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
g.write_edgelist(os.path.join(outdir, 'graph'))

xs = []
imps = []
means = []
res = {}
ps = [0.05, 0.1, 0.15, 0.3, 0.5, 0.7]
print 'Running {0}'.format(name)
print '#nodes =', g.vcount()
print '#edges =', g.ecount()
print 'transitivity =', g.transitivity_undirected()
for p in ps:
    niter = 100
    nsim_nonad = 1000
    nsim_ad = 100
    k_ratio = 20
    gamma = 1
    n_workers = 7
    r = influence.compare(g, p, nsim_nonad, nsim_ad, niter, k_ratio, gamma, n_workers)
    fm_nonad = np.mean(r['r_nonad_rg'])
    fm_ad = np.mean(r['r_ad'])
    print 'nonad =', fm_nonad
    print 'ad =', fm_ad
    imp = 100.0 * (fm_ad - fm_nonad) / fm_nonad
    means.append(imp)

ms = util.get_coords(ps, means)
texname = model.lower() + '.tex'
util.replace(outdir, FN_MC_IMP, {'means': ms}, outname=texname)
util.maketex(outdir, texname)

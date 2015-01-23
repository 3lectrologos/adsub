import os
import numpy as np
import igraph as ig
import util
import maxcut

FN_MC_IMP = 'mc_imp.tex'

model = 'SNAP_GR'
nodes = 1000
(name, g) = util.get_tc(model, nodes, directed=False)
ig.plot(g)
util.plot_degree_dist(g)

outdir = os.path.join(util.DIR_RES, 'maxcut', model)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
g.write_edgelist(os.path.join(outdir, 'graph'))

xs = []
imps = []
means = []
res = {}
n_available = len(g.vs) / 10
ks = [int(kr * n_available) for kr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]]
print 'Running {0}'.format(name)
print '#nodes =', g.vcount()
print '#edges =', g.ecount()
print 'transitivity =', g.transitivity_undirected()
for k in ks:
    reps = 5
    niter = 1000
    nsim_nonad = 1000
    r = maxcut.run(g, reps, niter, nsim_nonad, n_available, k)
    imps += r['imp']
    means.append(np.mean(r['imp']))
    xs += [k] * len(r['imp'])

ms = util.get_coords(ks, means)
cs = util.get_coords(xs, imps)
texname = model.lower() + '.tex'
util.replace(outdir, FN_MC_IMP, {'imp': cs, 'means': ms}, outname=texname)
util.maketex(outdir, texname)

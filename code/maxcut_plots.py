import numpy as np
import igraph as ig
import util
import maxcut

FN_MC_IMP = 'mc_imp.tex'

ks = [5, 10, 15, 20, 25]
xs = []
imps = []
means = []
res = {}
model = 'PROTEIN'
nodes = 1459
(name, g) = util.get_tc(model, nodes, directed=False)
ig.plot(g)
util.plot_degree_dist(g)
print 'Running {0}'.format(name)
print '#nodes =', g.vcount()
print '#edges =', g.ecount()
print 'transitivity =', g.transitivity_undirected()
for k in ks:
    reps = 5
    niter = 1000
    nsim_nonad = 1000
    n_available = 100
    r = maxcut.run(g, reps, niter, nsim_nonad, n_available, k)
    imps += r['imp']
    means.append(np.mean(r['imp']))
    xs += [k] * len(r['imp'])

ms = util.get_coords(ks, means)
cs = util.get_coords(xs, imps)
util.replace(util.DIR_RES, FN_MC_IMP, {'imp': cs, 'means': ms})
util.maketex(util.DIR_RES, FN_MC_IMP)

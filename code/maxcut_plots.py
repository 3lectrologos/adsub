import numpy as np
import util
import maxcut

FN_MC_IMP = 'mc_imp.tex'

ks = [1, 3, 5, 10, 15, 20]
xs = []
imps = []
means = []
res = {}
model = 'SNAP_ENRON'
nodes = 1000
(name, g) = util.get_tc(model, nodes, directed=False)
print 'Running {0}'.format(name)
print '#nodes =', g.vcount()
print '#edges =', g.ecount()
print 'transitivity =', g.transitivity_undirected()
for k in ks:
    reps = 10
    niter = 1000
    nsim_nonad = 1000
    n_available = 50
    r = maxcut.run(g, reps, niter, nsim_nonad, n_available, k)
    imps += r['imp']
    means.append(np.mean(r['imp']))
    xs += [k] * len(r['imp'])

ms = util.get_coords(ks, means)
cs = util.get_coords(xs, imps)
util.replace(util.DIR_RES, FN_MC_IMP, {'imp': cs, 'means': ms})
util.maketex(util.DIR_RES, FN_MC_IMP)

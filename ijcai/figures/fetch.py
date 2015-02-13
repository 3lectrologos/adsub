import cPickle as pcl
import os
import sys
sys.path.append(os.path.abspath('../../code/'))
import numpy as np
import igraph as ig
import util


TEMPLATE_FS = 'imp_ijcai.tex'
TEMPLATE_PS = 'pimp_ijcai.tex'
OUTDIR = '.'
MODELS = ['EGO_FB', 'GNUTELLA', 'GPLUS', 'TWITTER']
OBJ = ['influence', 'maxcut']
OBJ_SHORT = {'influence': 'inf', 'maxcut': 'mc', 'pinfluence': 'pinf', 'pmaxcut': 'pmc'}

def get_one(obj, model):
    lmodel = model.lower()
    resdir = os.path.join(util.DIR_RES, obj, model)
    with open(os.path.join(resdir, lmodel + '.pickle')) as f:
        data = pcl.load(f)
        g = ig.Graph.Read_Edgelist(os.path.join(resdir, 'graph'))
        cs = {}
        means = []
        stds = []
        ks = np.unique(data['ks'])
        for k in ks:
            reps = sum(data['ks'] == k)
            ys_ad = np.array(data['ad'])[data['ks'] == k]
            ys_nonad = np.array(data['nonad'])[data['ks'] == k]
            ys = [(100.0*(foo-bar))/foo for foo, bar in zip(ys_ad, ys_nonad)]
            means.append(np.mean(ys))
            stds.append(np.std(ys)/np.sqrt(reps))
        cs['imp'] = util.get_coords(ks, means, stds)
        if obj == 'influence' or obj == 'maxcut':
            template = TEMPLATE_FS
        else:
            template = TEMPLATE_PS
        texname = OBJ_SHORT[obj] + '_' + lmodel + '.tex'
        util.replace(OUTDIR, template, cs, texname)
        print '---------'

if __name__ == '__main__':
    for obj in ['influence', 'maxcut']:
        for model in MODELS:
            print obj, model
            get_one(obj, model)
    print 'pinfluence', 'EGO_FB'
    get_one('pinfluence', 'EGO_FB')
    print 'pmaxcut', 'GPLUS'
    get_one('pmaxcut', 'GPLUS')

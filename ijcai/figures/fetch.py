import cPickle as pcl
import os
import sys
sys.path.append(os.path.abspath('../../code/'))
import numpy as np
import igraph as ig
import util


TEMPLATE_FS = 'imp_ijcai.tex'
TEMPLATE_PS = 'pimp_ijcai.tex'
TEMPLATE_FAVG = 'fs_ijcai.tex'
OUTDIR = '.'
MODELS = ['EGO_FB', 'GNUTELLA', 'GPLUS', 'TWITTER']
OBJ_SHORT = {
    'influence': 'inf',
    'maxcut': 'mc',
    'pinfluence': 'pinf',
    'pmaxcut': 'pmc',
    'fsinfluence': 'fsinf',
    'fsmaxcut': 'fsmc'
}


def get_one(obj, model):
    lmodel = model.lower()
    if obj[:2] == 'fs':
        resdir = os.path.join(util.DIR_RES, obj[2:], model)
    else:
        resdir = os.path.join(util.DIR_RES, obj, model)
    with open(os.path.join(resdir, lmodel + '.pickle')) as f:
        data = pcl.load(f)
        g = ig.Graph.Read_Edgelist(os.path.join(resdir, 'graph'))
        cs = {}
        ks = np.unique(data['ks'])
        if obj[:2] != 'fs':
            means = []
            stds = []
            for k in ks:
                reps = sum(data['ks'] == k)
                ys_ad = np.array(data['ad'])[data['ks'] == k]
                ys_nonad = np.array(data['nonad'])[data['ks'] == k]
                ys = [(100.0*(foo-bar))/max(foo,bar)
                      for foo, bar in zip(ys_ad, ys_nonad)]
                means.append(np.mean(ys))
                stds.append(np.std(ys)/np.sqrt(reps))
            cs['imp'] = util.get_coords(ks, means, stds)
            if obj == 'influence' or obj == 'maxcut':
                template = TEMPLATE_FS
            else:
                template = TEMPLATE_PS
            texname = OBJ_SHORT[obj] + '_' + lmodel + '.tex'
            util.replace(OUTDIR, template, cs, texname)
        else:
            means_ad = []
            stds_ad = []
            means_nonad = []
            stds_nonad = []
            for k in ks:
                reps = sum(data['ks'] == k)
                ys_ad = np.array(data['ad'])[data['ks'] == k]
                means_ad.append(np.mean(ys_ad))
                stds_ad.append(np.std(ys_ad))
                ys_nonad = np.array(data['nonad'])[data['ks'] == k]
                means_nonad.append(np.mean(ys_nonad))
                stds_nonad.append(np.std(ys_nonad))
            cs['mean_ad'] = util.get_coords(ks, means_ad, stds_ad)
            cs['mean_nonad'] = util.get_coords(ks, means_nonad, stds_nonad)
            texname = OBJ_SHORT[obj] + '_' + lmodel + '.tex'
            util.replace(OUTDIR, TEMPLATE_FAVG, cs, texname)
        print '---------'


if __name__ == '__main__':
    # for obj in ['influence', 'maxcut']:
    #     for model in MODELS:
    #         print obj, model
    #         get_one(obj, model)
    # print 'pinfluence', 'EGO_FB'
    # get_one('pinfluence', 'EGO_FB')
    # print 'pmaxcut', 'TWITTER'
    # get_one('pmaxcut', 'TWITTER')
    print 'fsinfluence', 'EGO_FB'
    get_one('fsinfluence', 'EGO_FB')
    print 'fsmaxcut', 'EGO_FB'
    get_one('fsmaxcut', 'EGO_FB')

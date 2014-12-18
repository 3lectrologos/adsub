import os
import argparse
import cPickle as pcl
import subprocess
import numpy as np
import util


FN_FAVG = 'f_avg.tex'
FN_FRATIO = 'f_ratio.tex'
FN_FNODES = 'f_nodes.tex'

def process_dir(datadir):
    pedge = []
    f_nonad = []
    f_ad = []
    f_ratio = []
    f_nodes_nonad = []
    f_nodes_ad = []
    
    fnames = [s for s in os.listdir(datadir) if s.endswith('pickle')]

    for fname in fnames:
        with open(os.path.join(datadir, fname)) as f:
            print 'Processing', fname
            data = pcl.load(f)
            n = 0.01*data['vcount']
            pedge.append(data['pedge'])
            f_nonad.append(np.mean(data['r_nonad_rg']))
            f_ad.append(np.mean(data['r_ad']))
            f_ratio.append((np.mean(data['r_ad'])-np.mean(data['r_nonad_rg']))/n)
            print data['pedge'], ',', data['v_nonad_rg']
            f_nodes_nonad.append(np.mean(len(data['v_nonad_rg']))/n)
            f_nodes_ad.append(np.mean(data['v_ad'])/n)
            
    s_nonad = util.get_coords(pedge, f_nonad)
    s_ad = util.get_coords(pedge, f_ad)
    s_ratio = util.get_coords(pedge, f_ratio)
    s_nodes_nonad = util.get_coords(pedge, f_nodes_nonad)
    s_nodes_ad = util.get_coords(pedge, f_nodes_ad)
            
    util.replace(datadir, FN_FAVG, {'f_avg_nonad': s_nonad, 'f_avg_ad': s_ad})
    util.replace(datadir, FN_FRATIO, {'f_ratio': s_ratio})
    util.replace(datadir, FN_FNODES, {'f_nodes_nonad': s_nodes_nonad,
                                     'f_nodes_ad': s_nodes_ad})

    util.maketex(datadir, FN_FAVG)
    util.maketex(datadir, FN_FRATIO)
    util.maketex(datadir, FN_FNODES)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    for d in os.walk(util.DIR_DATA):
        if d[2] != []:
            datadir = os.path.join(util.DIR_DATA, d[0])
            if args.clean:
                util.clean(datadir)
            else:
                process_dir(datadir)

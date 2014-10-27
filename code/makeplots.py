import os
import argparse
import cPickle as pcl
import subprocess
import numpy as np


DIR_DATA = '../results'
DIR_TEMPLATE = '../templates'
FN_FAVG = 'f_avg.tex'
FN_FRATIO = 'f_ratio.tex'
FN_FNODES = 'f_nodes.tex'

def get_coords(x, y):
    s = ''
    z = sorted(zip(x, y))
    for e in z:
        s += '(' + str(e[0]) + ',' + str(e[1]) + ')'
    return s
    
def replace(datadir, fname, d):
    with open(os.path.join(DIR_TEMPLATE, fname), 'r') as fin:
        s = fin.read()
        for k in d:
            s = s.replace('%' + k + '%', d[k])
            with open(os.path.join(datadir, fname), 'w') as fout:
                fout.write(s)

def maketex(datadir, fname):
    cwd = os.getcwd()
    os.chdir(datadir)
    subprocess.call(['latexmk', '-pdf', fname])
    subprocess.call(['latexmk', '-c'])
    os.chdir(cwd)

def clean(datadir):
    cwd = os.getcwd()
    os.chdir(datadir)
    subprocess.call('rm -f *.tex', shell=True)
    os.chdir(cwd)

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
            
    s_nonad = get_coords(pedge, f_nonad)
    s_ad = get_coords(pedge, f_ad)
    s_ratio = get_coords(pedge, f_ratio)
    s_nodes_nonad = get_coords(pedge, f_nodes_nonad)
    s_nodes_ad = get_coords(pedge, f_nodes_ad)
            
    replace(datadir, FN_FAVG, {'f_avg_nonad': s_nonad, 'f_avg_ad': s_ad})
    replace(datadir, FN_FRATIO, {'f_ratio': s_ratio})
    replace(datadir, FN_FNODES, {'f_nodes_nonad': s_nodes_nonad,
                                 'f_nodes_ad': s_nodes_ad})

    maketex(datadir, FN_FAVG)
    maketex(datadir, FN_FRATIO)
    maketex(datadir, FN_FNODES)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    for d in os.walk(DIR_DATA):
        if d[2] != []:
            datadir = os.path.join(DIR_DATA, d[0])
            if args.clean:
                clean(datadir)
            else:
                process_dir(datadir)

import os
import cPickle as pcl
import subprocess
import numpy as np


DIR_DATA = '../results/SNAP_GR'
DIR_TEMPLATE = '../templates'
FN_FAVG = 'f_avg.tex'
FN_FRATIO = 'f_ratio.tex'

def get_coords(x, y):
    s = ''
    z = sorted(zip(x, y))
    for e in z:
        s += '(' + str(e[0]) + ',' + str(e[1]) + ')'
    return s
    
def replace(fname, d):
    with open(ftemp(fname), 'r') as fin:
        s = fin.read()
        for k in d:
            s = s.replace('%' + k + '%', d[k])
            with open(fdata(fname), 'w') as fout:
                fout.write(s)

def ftemp(fname):
    return os.path.join(DIR_TEMPLATE, fname)

def fdata(fname):
    return os.path.join(DIR_DATA, fname)

def maketex(fname):
    cwd = os.getcwd()
    os.chdir(DIR_DATA)
    subprocess.call(['latexmk', '-pdf', fname])
    subprocess.call(['latexmk', '-c'])
    os.chdir(cwd)

pedge = []
f_nonad = []
f_ad = []
f_ratio = []

fnames = [s for s in os.listdir(DIR_DATA) if s.endswith('pickle')]

for fname in fnames:
    with open(os.path.join(DIR_DATA, fname)) as f:
        data = pcl.load(f)
        n = 0.01*data['vcount']
        pedge.append(data['pedge'])
        f_nonad.append(np.mean(data['r_nonad_rg']))
        f_ad.append(np.mean(data['r_ad']))
        f_ratio.append((np.mean(data['r_ad'])-np.mean(data['r_nonad_rg']))/n)


s_nonad = get_coords(pedge, f_nonad)
s_ad = get_coords(pedge, f_ad)
s_ratio = get_coords(pedge, f_ratio)

print s_ratio

replace(FN_FAVG, {'f_avg_nonad': s_nonad, 'f_avg_ad': s_ad})
replace(FN_FRATIO, {'f_ratio': s_ratio})

maketex(FN_FAVG)
maketex(FN_FRATIO)

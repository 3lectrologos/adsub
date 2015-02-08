#!/usr/bin/env python
import os
import igraph as ig
import util

files = os.listdir(util.DIR_DATA)
print 'Total number of data sets:', len(files)
print ''
for m in files:
    print '=================='
    print m
    g = ig.Graph.Read_Edgelist(os.path.join(util.DIR_DATA, m))
    print '#nodes:', g.vcount()
    print '#edges:', g.ecount()
    print '==================\n'

#!/usr/bin/env python
import argparse
import os
import util

models = [os.path.splitext(x)[0] for x in os.listdir(util.DIR_DATA)]

parser = argparse.ArgumentParser(
    description='Adaptive monotone submodular experiments')
parser.add_argument('objective',
                    choices=['inf', 'mc'],
                    help='objective function')
parser.add_argument('-s', '--slow',
                    action='store_true',
                    help='more detailed simulation')
args = parser.parse_args()

n = 1000
for m in models:
    cargs = [args.objective, m, str(n)]
    if args.slow:
        cargs = ['-s'] + cargs
    cmd = 'python plots.py ' + ' '.join(cargs)
    print cmd
    os.system(cmd)

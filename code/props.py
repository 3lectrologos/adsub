#!/usr/bin/env python
import argparse
import os
import numpy as np
import scipy.linalg as spl
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import matplotlib.pylab as plt
import igraph as ig
import util


PROP_NAMES = [
    'Avg. local transitivity',
    'Global transitivity',
    'Diameter',
    'Degree assortativity',
    'Avg. path length',
    'Max. clique size',
    'Density',
    'Max. degree',
    'Median degree',
    'Power law slope',
    'Spectral norm',
    'Laplacian 2nd smallest eig.'
    ]


def get_graph_props(g):
    travgl = g.transitivity_avglocal_undirected()
    tru = g.transitivity_undirected()
    d = g.diameter()
    asd = g.assortativity_degree()
    apl = g.average_path_length()
    omega = g.omega()
    density = g.density()
    maxd = g.maxdegree()
    medd = np.median(g.degree())
    plaw = ig.power_law_fit(g.degree())
    spnorm = max(np.abs(spl.eigvals(g.get_adjacency().data)))
    leigs = spl.eigvals(g.laplacian())
    algc = abs(sorted([e for e in leigs if e > 1e-10])[1])
    
    return [travgl,     # avg. local transitivity
            tru,        # global transitivity
            d,          # diameter
            asd,        # degree assortativity
            apl,        # avg. path length
            omega,      # max. clique
            density,
            maxd,       # max. degree
            medd,       # median degree
            plaw.alpha, # power law exponent
            spnorm,     # largest eigenvalue of adj. matrix
            algc,       # 2nd smallest non-zero eigenvalue of laplacian
            ]


def get_features(gdir):
    X = []
    labels = []
    subdirs = sorted(os.listdir(gdir))
    for subdir in ['EGO_FB', 'GNUTELLA', 'GPLUS', 'TWITTER']:#subdirs:
        if not os.path.isdir(os.path.join(gdir, subdir)):
            continue
        with open(os.path.join(gdir, subdir, 'props'), 'r') as f:
            X.append([float(p) for p in f.readlines()])
        labels.append(subdir)
    return (X, labels)


def save_features(gdir):
    subdirs = sorted(os.listdir(gdir))
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(gdir, subdir)):
            continue
        print subdir
        g = ig.Graph.Read_Edgelist(os.path.join(gdir, subdir, 'graph'),
                                   directed=False)
        props = get_graph_props(g)
        with open(os.path.join(gdir, subdir, 'props'), 'w') as f:
            for p in props:
                f.write(str(p) + '\n')


def train(X, Y):
#    clf = ske.RandomForestClassifier()
    clf = skl.LogisticRegression(C=10)
    X_new = clf.fit(X, Y).transform(X)
    print X_new


def train_2d(X, Y, labels, fs, plot=True):
    X = X[:, fs]
    clf = skl.LogisticRegression(C=10)
    clf.fit(X, Y)
#    clf = ske.RandomForestClassifier()
    if plot:
        h = 0.1
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.plot(X[Y==1, 0], X[Y==1, 1], 'o',
                 markerfacecolor=plt.cm.Paired(0.5),
                 markersize=10)
        plt.plot(X[Y==0, 0], X[Y==0, 1], 'o',
                 markerfacecolor=plt.cm.Paired(0.1),
                 markersize=10)
#    labels = ['a' + str(i) for i in range(X.shape[0])]
        for label, x, y in zip(labels, X[:, 0], X[:, 1]):
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(-10, 10),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         fontsize=6,
                         bbox={'boxstyle': 'round,pad=0.5', 'fc': '#DDDDDD',
                               'alpha': 0.3},
                         arrowprops={'arrowstyle': '->',
                                     'connectionstyle': 'arc3,rad=0'})
        plt.xlabel(PROP_NAMES[fs[0]])
        plt.ylabel(PROP_NAMES[fs[1]])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()
    return clf.score(X, Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance classifier')
    parser.add_argument('-g', '--generate',
                        action='store_true',
                        help='generate graph features')
    parser.add_argument('-o', '--optimize',
                        action='store_true',
                        help='optimize over pairs of features')
    parser.add_argument('objective',
                        choices=['inf', 'mc'],
                        help='objective function')
    args = parser.parse_args()
    if args.objective == 'inf':
        subdir = 'influence'
        Y = [1, 0, 0, 1]
#        Y = [0, 1, 1, 0, 1, 0,
#             1, 0, 0, 1, 1, 0,
#             1, 0, 0, 0, 0, 1,
#             1, 1, 1, 1, 0, 1,
#             1, 0]
    elif args.objective == 'mc':
        subdir = 'maxcut'
        Y = [1, 0, 1, 0]
#        Y = [0, 0, 1, 1, 0, 0,
#             0, 1, 0, 0, 0, 0,
#             0, 0, 1, 0, 0, 1,
#             1, 0, 0, 0, 1, 0,
#             1, 1]
    if args.generate:
        save_features(os.path.join(util.DIR_RES, subdir))
    (X, labels) = get_features(os.path.join(util.DIR_RES, subdir))
    X = np.array(X)
    Y = np.array(Y).ravel()
    if args.optimize:
        scores = []
        for r in range(X.shape[1]):
            for s in range(r+1, X.shape[1]):
                score = train_2d(X, Y, labels, [r, s], plot=False)
                scores.append(((r, s), score))
                print r, s, ':', score
        m = max(scores, key=lambda x: x[1])
        print 'best:', [x[0] for x in scores if x[1] == m[1]]
        print 'score:', m[1]
    else:
        train_2d(X, Y, labels, [2, 7])

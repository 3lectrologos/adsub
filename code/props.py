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
    'Avg. shortest path length',
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
    aspl = np.mean(g.shortest_paths())
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
            aspl,       # avg. shortest path length
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
#    Y = []
    labels = []
    subdirs = sorted(os.listdir(gdir))
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(gdir, subdir)):
            continue
        print subdir
        with open(os.path.join(gdir, subdir, 'props'), 'r') as f:
            X.append([float(p) for p in f.readlines()])
#        with open(os.path.join(gdir, subdir, 'res'), 'r') as f:
#            Y.append([int(f.readline())])
        labels.append(subdir)
    Y = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    return (X, Y, labels)


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


def train_2d(X, Y, labels, fs):
    print fs
    print X[:, fs]
    X = X[:, fs]

    clf = skl.LogisticRegression(C=10)
#    clf = ske.RandomForestClassifier()
    clf.fit(X, Y)

    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.plot(X[Y==1, 0], X[Y==1, 1], 'o',
             markerfacecolor=plt.cm.Paired(0.5),
             markersize=10,
             antialiased=True)
    plt.plot(X[Y==0, 0], X[Y==0, 1], 'o',
             markerfacecolor=plt.cm.Paired(0.1),
             markersize=10,
             antialiased=True)
#    labels = ['a' + str(i) for i in range(X.shape[0])]
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(label,
                     xy = (x, y), xytext = (-20, 20),
                     textcoords = 'offset points', ha = 'right', va = 'bottom',
                     bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.xlabel(PROP_NAMES[fs[0]])
    plt.ylabel(PROP_NAMES[fs[1]])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    


if __name__ == '__main__':
    save_features(os.path.join(util.DIR_RES, 'maxcut'))
    (X, Y, labels) = get_features(os.path.join(util.DIR_RES, 'maxcut'))
    X = np.array(X)
    Y = np.array(Y).ravel()
    print X
#    train(X, Y)
    train_2d(X, Y, labels, [3, 10])

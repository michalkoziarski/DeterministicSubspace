import os
import sys
import cPickle as pickle

from datasets import *
from mutual_info import mutual_information_2d
from sklearn.cross_validation import StratifiedKFold


ROOT_PATH = 'mutual_information'
CV = 2
N_ITERATIONS = 5


def precalculate(X, y, cv=CV):
    folds = []

    for _ in range(N_ITERATIONS):
        skf = StratifiedKFold(y, cv, shuffle=True)

        for train, test in skf:
            X_train = X[train]
            mutual_information = [[0 for _ in range(X_train.shape[1])] for _ in range(X_train.shape[1])]

            for i in range(X_train.shape[1]):
                for j in range(i, X_train.shape[1]):
                    mi = mutual_information_2d(X_train[:, i], X_train[:, j], normalized=True)
                    mutual_information[i][j] = mi
                    mutual_information[j][i] = mi

            folds.append((train, test, mutual_information))

    return folds


def save(folds, fname):
    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)

    with open(os.path.join(ROOT_PATH, fname), 'w') as f:
        pickle.dump(folds, f)


def load(fname):
    with open(os.path.join(ROOT_PATH, fname), 'r') as f:
        folds = pickle.load(f)

    return folds


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            dataset = globals()['load_' + sys.argv[1]]()
        except:
            dataset = globals()['load_keel'](sys.argv[1])

        datasets = {sys.argv[1]: dataset}
    else:
        datasets = load_all()

    for dataset_name, dataset in datasets.iteritems():
        X, y = dataset
        folds = precalculate(X, y, CV)
        save(folds, dataset_name)

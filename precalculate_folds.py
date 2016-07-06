import cPickle as pickle

from datasets import load_all
from sklearn.cross_validation import KFold


datasets = load_all()
folds = {}

for name, dataset in datasets.iteritems():
    folds[name] = []
    X, y = dataset

    for _ in range(5):
        for train_idx, test_idx in KFold(len(y), n_folds=2, shuffle=True):
            folds[name].append((train_idx, test_idx))

pickle.dump(folds, open('folds.pickle', 'wb'))

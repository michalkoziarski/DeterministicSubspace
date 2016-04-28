import sys

from subspace import *
from datasets import *
from precalculate_mi import load
from time import gmtime, strftime
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


if len(sys.argv) > 1:
    try:
        dataset = globals()['load_' + sys.argv[1]]()
    except:
        dataset = globals()['load_keel'](sys.argv[1])

    datasets = {sys.argv[1]: dataset}
else:
    datasets = load_all()

if len(sys.argv) > 2:
    K = [int(sys.argv[2])]
else:
    K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVM': LinearSVC(),
    'NaiveBayes': GaussianNB()
}


def test(X, y, train, test, clf, dataset_name, clf_name, k, method='-', alpha='-', b='-', omega='-', date=None):
    file_name = '%s_%s_k_%d.csv' % (dataset_name, date, k)

    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists(os.path.join('results', file_name)):
        with open(os.path.join('results', file_name), 'w') as f:
            f.write('dataset, method, classifier, k, alpha, b, omega, accuracy\n')

    score = clone(clf).fit(X[train], y[train]).score(X[test], y[test])

    with open(os.path.join('results', file_name), 'a') as f:
        f.write('%s, %s, %s, %d, %s, %s, %s, %.3f\n' % (dataset_name, method, clf_name, k, alpha, b, omega, score))

    print 'Dataset: %s, method: %s, classifier: %s, k: %d, alpha: %s, b: %s, omega: %s, accuracy: %.3f' % \
          (dataset_name, method, clf_name, k, alpha, b, omega, score)


for dataset_name, dataset in datasets.iteritems():
    date = strftime('%Y_%m_%d_%H-%M-%S', gmtime())
    X, y = dataset
    n = X.shape[1] / 2
    folds = load(dataset_name)

    for train, test, mutual_information in folds:
        for k in K:
            test(X, y, train, test, RandomForestClassifier(n_estimators=k), dataset_name,
                 'RandomForest', k, date=date)

            for clf_name, clf in classifiers.iteritems():
                test(X, y, train, test, RandomSubspaceClassifier(clf, k=k, n=n), dataset_name,
                     clf_name, k, 'RandomSubspace', date=date)

                for alpha in [0., 0.25, 0.5, 0.75, 0.95]:
                    for b in [1, 2, 3]:
                        for omega in [1.]:
                            test(X, y, train, test, DeterministicSubspaceClassifier(clf, k=k, n=n, b=b,
                                 alpha=alpha, omega=omega, mutual_information=mutual_information),
                                 dataset_name, clf_name, k, 'DeterministicSubspace', alpha, b, omega, date=date)

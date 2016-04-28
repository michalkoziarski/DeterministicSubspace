import os
import sys

from subspace import *
from datasets import *
from mutual_info import mutual_information_2d
from time import gmtime, strftime
from sklearn import cross_validation
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

classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVM': LinearSVC(),
    'NaiveBayes': GaussianNB()
}


def test(X, y, clf, dataset_name, clf_name, k, method='-', alpha='-', b='-', omega='-', cv=5, date=None):
    if date:
        file_name = '%s_%s.csv' % (dataset_name, date)
    else:
        file_name = '%s.csv' % dataset_name

    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists(os.path.join('results', file_name)):
        with open(os.path.join('results', file_name), 'w') as f:
            f.write('dataset, method, classifier, k, alpha, b, omega, accuracy\n')

    score = cross_validation.cross_val_score(clf, X, y, cv=5).mean()

    with open(os.path.join('results', file_name), 'a') as f:
        f.write('%s, %s, %s, %d, %s, %s, %s, %.3f\n' % (dataset_name, method, clf_name, k, alpha, b, omega, score))

    print 'Dataset: %s, method: %s, classifier: %s, k: %d, alpha: %s, b: %s, omega: %s, accuracy: %.3f' % (dataset_name, method, clf_name, k, alpha, b, omega, score)


for dataset_name, dataset in datasets.iteritems():
    date = strftime('%Y_%m_%d_%H-%M-%S', gmtime())
    X, y = dataset
    n = X.shape[1] / 2

    mutual_information = [[0 for i in range(X.shape[1])] for j in range(X.shape[1])]

    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            mi = mutual_information_2d(X[:, i], X[:, j], normalized=True)
            mutual_information[i][j] = mi
            mutual_information[j][i] = mi

    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        test(X, y, RandomForestClassifier(n_estimators=k), dataset_name,
             'RandomForest', k, date=date)

        for clf_name, clf in classifiers.iteritems():
            test(X, y, RandomSubspaceClassifier(clf, k=k, n=n), dataset_name,
                 clf_name, k, 'RandomSubspace', date=date)

            for alpha in [0., 0.25, 0.5, 0.75, 1.]:
                for b in [1, 2, 3]:
                    for omega in [1.]:
                        test(X, y, DeterministicSubspaceClassifier(clf, k=k, n=n, b=b,
                             alpha=alpha, omega=omega, mutual_information=mutual_information),
                             dataset_name, clf_name, k, 'DeterministicSubspace', alpha, b, omega, date=date)

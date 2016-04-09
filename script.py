import os

from subspace import *
from datasets import load_all
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


datasets = load_all()

classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVM': LinearSVC(),
    'NaiveBayes': GaussianNB()
}


def test(X, y, clf, dataset_name, clf_name, k, method='-', alpha='-', b='-', omega='-', cv=5, file_name='log.csv'):
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
    X, y = dataset
    n = X.shape[1] / 2

    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        test(X, y, RandomForestClassifier(n_estimators=k), dataset_name,
             'RandomForest', k)

        for clf_name, clf in classifiers.iteritems():
            test(X, y, RandomSubspaceClassifier(clf, k=k, n=n), dataset_name,
                 clf_name, k, 'RandomSubspace')

            for alpha in [0., 0.25, 0.5, 0.75, 1.]:
                for b in [1]:
                    for omega in [6.]:
                        test(X, y, DeterministicSubspaceClassifier(clf, k=k, n=n, b=b, alpha=alpha, omega=omega),
                             dataset_name, clf_name, k, 'DeterministicSubspace', alpha, b, omega)

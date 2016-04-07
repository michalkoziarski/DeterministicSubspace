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


def test(X, y, clf, dataset_name, clf_name, k, cv=5, file_name='log.csv'):
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists(os.path.join('results', file_name)):
        with open(os.path.join('results', file_name), 'w') as f:
            f.write('classifier, dataset, k, accuracy\n')

    score = cross_validation.cross_val_score(clf, X, y, cv=5).mean()

    with open(os.path.join('results', file_name), 'a') as f:
        f.write('%s, %s, %d, %.3f\n' % (clf_name, dataset_name, k, score))

    print 'Classifier: %s, dataset: %s, k: %d, score: %.3f' % (clf_name, dataset_name, k, score)


for dataset_name, dataset in datasets.iteritems():
    X, y = dataset
    n = X.shape[1] / 2

    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        test(X, y, RandomForestClassifier(n_estimators=k), dataset_name,
             'RandomForest', k)

        for clf_name, clf in classifiers.iteritems():
            test(X, y, RandomSubspaceClassifier(clf, k=k, n=n), dataset_name,
                 'RandomSubspace#%s' % clf_name, k)

            test(X, y, DeterministicSubspaceClassifier(clf, k=k, n=n),
                 dataset_name, 'DeterministicSubspace#%s' % clf_name, k)

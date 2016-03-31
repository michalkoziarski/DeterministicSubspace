from subspace import *
from datasets import load_all
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier


datasets = load_all()

classifiers = {
    'DecisionTree': DecisionTreeClassifier()
}


for dataset_name, dataset in datasets.iteritems():
    for clf_name, clf in classifiers.iteritems():
        for k in [5]:
            print dataset_name
            X, y = dataset
            n = X.shape[1] / 2

            scores = cross_validation.cross_val_score(
                RandomSubspaceClassifier(clf, k=k, n=n), X, y, cv=5)
            print scores.mean()

            scores = cross_validation.cross_val_score(
                DeterministicSubspaceClassifier(clf, k=k, n=n), X, y, cv=5)
            print scores.mean()

            scores = cross_validation.cross_val_score(
                GaussianSubspaceClassifier(clf, k=k, n=n, b=2 * k), X, y, cv=5)
            print scores.mean()

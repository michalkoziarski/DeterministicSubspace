from subspace import *
from datasets import load_all
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


datasets = load_all()

classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVM': LinearSVC(),
    'NaiveBayes': GaussianNB()
}


for dataset_name, dataset in datasets.iteritems():
    for clf_name, clf in classifiers.iteritems():
        for k in [5]:
            print 'Dataset: %s, base classifier: %s' % (dataset_name, clf_name)

            X, y = dataset
            n = X.shape[1] / 2

            scores = cross_validation.cross_val_score(
                RandomSubspaceClassifier(clf, k=k, n=n), X, y, cv=5)
            print 'Random Subspace: %.3f' % scores.mean()

            scores = cross_validation.cross_val_score(
                MutualInformationRoundRobinSubspaceClassifier(clf, k=k, n=n), X, y, cv=5)
            print 'Deterministic Subspace: %.3f' % scores.mean()

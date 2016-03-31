import numpy as np
import pandas as pd

from subspace import *
from sklearn.tree import DecisionTreeClassifier


matrix = pd.read_csv('data/spambase.csv', header=None).as_matrix()

random_scores = []
gaussian_scores = []

for _ in range(10):
    idx = np.random.permutation(len(matrix))

    X, y = matrix[idx, :-1], matrix[idx, -1]

    X_train, y_train = X[:len(matrix) / 2], y[:len(matrix) / 2]
    X_test, y_test = X[len(matrix) / 2:], y[len(matrix) / 2:]

    clf = RandomSubspaceClassifier(DecisionTreeClassifier(), k=5, n=3)
    #clf = DeterministicSubspaceClassifier(DecisionTreeClassifier(), k=9, n=3, alpha=0.5)
    clf.fit(X_train, y_train)
    #print clf.score(X_test, y_test)
    random_scores.append(clf.score(X_test, y_test))

    clf = GaussianSubspaceClassifier(DecisionTreeClassifier(), k=5, n=3, b=15)
    clf.fit(X_train, y_train)
    #print clf.score(X_test, y_test)
    gaussian_scores.append(clf.score(X_test, y_test))

print np.mean(random_scores)
print np.mean(gaussian_scores)

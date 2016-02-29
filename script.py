import numpy as np
import pandas as pd

from deterministic_subspace import DeterministicSubspaceClassifier
from random_subspace import RandomSubspaceClassifier
from sklearn.tree import DecisionTreeClassifier


matrix = pd.read_csv('sonar.csv', header=None).as_matrix()
idx = np.random.permutation(len(matrix))

X, y = matrix[idx, :-1], matrix[idx, -1]

X_train, y_train = X[:len(matrix) / 2], y[:len(matrix) / 2]
X_test, y_test = X[len(matrix) / 2:], y[len(matrix) / 2:]

clf = DeterministicSubspaceClassifier(DecisionTreeClassifier(), k=9, n=3, alpha=0.5)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

clf = RandomSubspaceClassifier(DecisionTreeClassifier(), k=9, n=3)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

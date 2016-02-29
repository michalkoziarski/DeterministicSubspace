import numpy as np

from base_subspace import BaseSubspaceClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder


class RandomSubspaceClassifier(BaseSubspaceClassifier):
    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        self.clusters = []
        self.clfs = []

        for _ in range(self.k):
            self.clusters.append(np.random.permutation(range(X.shape[1]))[:self.n])

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

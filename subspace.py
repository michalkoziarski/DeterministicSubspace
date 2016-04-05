import math
import numpy as np

from sklearn import cross_validation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class BaseSubspaceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, k, n):
        """
        :param base_clf: base classifier
        :param k: number of clusters
        :param n: number of elements in a single cluster
        """
        self.base_clf = base_clf
        self.k = k
        self.n = n

        self.label_encoder = None
        self.clusters = None
        self.clfs = None

    def predict(self, X):
        predictions = np.asarray([self.clfs[i].predict(X[:, self.clusters[i]]) for i in range(len(self.clfs))],
                                 dtype=np.int8).T
        majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)

        return self.label_encoder.inverse_transform(majority)


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


class RoundRobinSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, alpha=0.5):
        """
        :param alpha: score (inside cluster) coefficient
        :param 1 - alpha: diversification (between clusters) coefficient
        """
        self.alpha = alpha

        super(RoundRobinSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.clusters = [[] for _ in range(self.k)]

        for _ in range(self.n):
            for index in range(self.k):
                cluster = self.clusters[index]
                scores = []

                for feature in range(X.shape[1]):
                    if feature in cluster:
                        score = -np.inf
                    else:
                        score = self._score(X, y, cluster, feature, index)

                    scores.append(score)

                cluster.append(np.argmax(scores))

        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _score(self, X, y, cluster, feature, index):
        return self.alpha * self._inside_score(X, y, cluster, feature) + \
               (1. - self.alpha) * self._outside_score(cluster, feature, index)

    def _inside_score(self, X, y, cluster, feature, cv=2):
        extended_cluster = cluster + [feature]

        return cross_validation.cross_val_score(self.base_clf, X[:, extended_cluster], y, cv=cv).mean()

    def _outside_score(self, cluster, feature, index):
        count = 0.
        total = 1e-9
        extended_cluster = cluster + [feature]

        for i in range(self.k):
            if i == index:
                continue

            current_cluster = self.clusters[i]
            total += len(current_cluster)

            for current_feature in current_cluster:
                if current_feature in extended_cluster:
                    count += 1

        return 1. - (count / total)


class PruningSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, b):
        """
        :param b: number of clusters before pruning
        """

        assert b >= k

        self.b = b

        super(PruningSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.clusters = [[] for _ in range(self.b)]

        for _ in range(self.n):
            for index in range(self.b):
                cluster = self.clusters[index]
                scores = []

                for feature in range(X.shape[1]):
                    if feature in cluster:
                        score = -np.inf
                    else:
                        score = self._inside_score(X, y, cluster, feature, index)

                    scores.append(score)

                cluster.append(np.argmax(scores))

        selected_clusters = []

        for _ in range(self.k):
            scores = []

            for index in range(self.b):
                cluster = self.clusters[index]

                if cluster in selected_clusters:
                    score = -np.inf
                else:
                    score = self._outside_score(cluster, selected_clusters)

                scores.append(score)

            selected_clusters.append(self.clusters[np.argmax(scores)])

        self.clusters = selected_clusters
        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _inside_score(self, X, y, cluster, feature, cv=2):
        extended_cluster = cluster + [feature]

        return cross_validation.cross_val_score(self.base_clf, X[:, extended_cluster], y, cv=cv).mean()

    def _outside_score(self, cluster, selected_clusters):
        count = 0.
        total = 1e-9

        for selected_cluster in selected_clusters:
            if cluster == selected_cluster:
                continue

            total += len(selected_cluster)

            for feature in selected_cluster:
                if feature in cluster:
                    count += 1

        return 1. - (count / total)

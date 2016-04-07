import math
import numpy as np
import mutual_info as mi

from sklearn import cross_validation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import StratifiedKFold
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


class DeterministicSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, alpha=0.5, b=3, omega=6.):
        """
        :param alpha: score (inside cluster) coefficient
        :param 1 - alpha: diversification (between clusters) coefficient
        :param b: number of clusters (multiplier of k) before pruning
        :param omega: multiplier used with error function
        """
        assert b >= 1

        self.alpha = alpha
        self.b = b
        self.omega = omega

        super(DeterministicSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.clusters = [[] for _ in range(int(self.k * self.b))]

        for _ in range(self.n):
            for index in range(int(self.k * self.b)):
                cluster = self.clusters[index]
                scores = []

                for feature in range(X.shape[1]):
                    if feature in cluster:
                        score = -np.inf
                    else:
                        score = score = self._score(X, y, cluster, feature, index)

                    scores.append(score)

                cluster.append(np.argmax(scores))

        if self.b > 1:
            selected_clusters = []

            for _ in range(self.k):
                scores = []
                counts, skf = self._correct_prediction_count(X, y, selected_clusters)

                for cluster in self.clusters:
                    if cluster in selected_clusters:
                        scores.append(-np.inf)
                    else:
                        scores.append(self._pruning_score(X, y, cluster, selected_clusters, counts, skf))

                selected_clusters.append(self.clusters[np.argmax(scores)])

            self.clusters = selected_clusters

        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _score(self, X, y, cluster, feature, index):
        return self.alpha * self._inside_score(X, y, cluster, feature) + \
               (1. - self.alpha) * self._outside_score(cluster, feature, index)

    def _inside_score(self, X, y, cluster, feature):
        if not hasattr(self, 'mutual_information'):
            self.mutual_information = [[0 for i in range(X.shape[1])] for j in range(X.shape[1])]

            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    mutual_information = mi.mutual_information_2d(X[:, i], X[:, j], normalized=True)
                    self.mutual_information[i][j] = mutual_information
                    self.mutual_information[j][i] = mutual_information

        if len(cluster) > 0:
            return np.min([self.mutual_information[feature][c] for c in cluster])
        else:
            return mi.mutual_information_2d(X[:, feature], y, normalized=True)

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

    def _correct_prediction_count(self, X, y, selected_clusters, cv=2):
        skf = StratifiedKFold(y, cv)
        counts = [0 for _ in range(len(y))]

        for cluster in selected_clusters:
            current_index = 0

            for train, test in skf:
                clf = clone(self.base_clf)
                clf.fit(X[train][:, cluster], y[train])
                predictions = clf.predict(X[test][:, cluster])

                for prediction, truth in zip(predictions, y[test]):
                    if prediction == truth:
                        counts[current_index] += 1

                    current_index += 1

        return counts, skf

    def _pruning_score(self, X, y, cluster, selected_clusters, counts, skf):
        counts = list(counts)
        unique = len(set(y))
        threshold = math.ceil(unique / 2.) / unique
        current_index = 0
        score = 0.

        for train, test in skf:
            clf = clone(self.base_clf)
            clf.fit(X[train][:, cluster], y[train])
            predictions = clf.predict(X[test][:, cluster])

            for prediction, truth in zip(predictions, y[test]):
                if prediction == truth:
                    counts[current_index] += 1

                current_index += 1

        for count in counts:
            score += math.erf((count / float(len(selected_clusters) + 1) - threshold) * self.omega)

        return score

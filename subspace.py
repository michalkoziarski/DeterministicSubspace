import math
import numpy as np
import mutual_info as mi

from collections import Counter
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


class BinaryMaskSubspaceClassifier(BaseSubspaceClassifier):
    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        self.clusters = []
        self.clfs = []

        mask = []
        starting_position = 0

        for i in range(X.shape[1]):
            column = []

            for j in range(self.k):
                if starting_position <= j < starting_position + self.k / 2:
                    column.append(1)
                else:
                    column.append(0)

            mirrored = [int(not x) for x in column]

            mask.append(column)
            mask.append(mirrored)

        # TODO : finish cluster initialization

        for _ in range(self.k):
            self.clusters.append(np.random.permutation(range(X.shape[1]))[:self.n])

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self


class DeterministicSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, alpha=0.5, b=2, omega=1., mutual_information=None):
        """
        :param alpha: score (inside cluster) coefficient
        :param 1 - alpha: diversification (between clusters) coefficient
        :param b: number of clusters (multiplier of k) before pruning
        :param omega: multiplier used with error function
        :param mutual_information: precalculated mutual information between every feature
        """
        assert b >= 1

        self.alpha = alpha
        self.b = b
        self.omega = omega
        self.mutual_information = mutual_information

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
                        score = self._score(X, y, cluster, feature, index)

                    scores.append(score)

                cluster.append(np.argmax(scores))

        if self.b > 1:
            selected_clusters = []

            for _ in range(self.k):
                scores = []
                counts, thresholds, skf = self._correct_prediction_count(X, y, selected_clusters)

                for cluster in self.clusters:
                    if cluster in selected_clusters:
                        scores.append(-np.inf)
                    else:
                        scores.append(self._pruning_score(X, y, cluster, selected_clusters, counts, thresholds, skf))

                selected_clusters.append(self.clusters[np.argmax(scores)])

            self.clusters = selected_clusters

        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _score(self, X, y, cluster, feature, index):
        return self.alpha * self._inside_score(X, y, cluster, feature, index) + \
               (1. - self.alpha) * self._outside_score(X, y, cluster, feature, index)

    def _inside_score(self, X, y, cluster, feature, index):
        if not hasattr(self, 'mutual_information'):
            self.mutual_information = [[0 for _ in range(X.shape[1])] for _ in range(X.shape[1])]

            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    mutual_information = mi.mutual_information_2d(X[:, i], X[:, j], normalized=True)
                    self.mutual_information[i][j] = mutual_information
                    self.mutual_information[j][i] = mutual_information

        if len(cluster) > 0:
            return np.min([self.mutual_information[feature][c] for c in cluster])
        else:
            return mi.mutual_information_2d(X[:, feature], y, normalized=True)

    def _outside_score(self, X, y, cluster, feature, index):
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
        predictions = []
        counts = []
        thresholds = []

        for cluster in selected_clusters:
            cluster_predictions = np.array([])

            for train, test in skf:
                clf = clone(self.base_clf)
                clf.fit(X[train][:, cluster], y[train])
                cluster_predictions = np.append(cluster_predictions, clf.predict(X[test][:, cluster]))

            predictions.append(cluster_predictions)

        if len(predictions) > 0:
            predictions = np.vstack(predictions)

        for i in range(len(y)):
            if len(predictions) > 0:
                counter = Counter(predictions[:, i])
                counts.append(counter[y[i]] if y[i] in counter.keys() else 0)

                if len(counter) > 1:
                    sorted_values = sorted(counter.values())
                    thresholds.append((math.floor((sorted_values[0] + sorted_values[1]) / 2.) + 1) / len(set(y)))
                else:
                    thresholds.append(0.5)
            else:
                counts.append(0)
                thresholds.append(1.)

        return counts, thresholds, skf

    def _pruning_score(self, X, y, cluster, selected_clusters, counts, thresholds, skf):
        counts = list(counts)
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

        for i in range(len(counts)):
            count = counts[i]
            threshold = thresholds[i]
            score += math.erf((count / float(len(selected_clusters) + 1) - threshold) * self.omega)

        return score


class MIDeterministicSubspaceClassifier(DeterministicSubspaceClassifier):
    def _outside_score(self, X, y, cluster, feature, index):
        if not hasattr(self, 'mutual_information'):
            self.mutual_information = [[0 for _ in range(X.shape[1])] for _ in range(X.shape[1])]

            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    mutual_information = mi.mutual_information_2d(X[:, i], X[:, j], normalized=True)
                    self.mutual_information[i][j] = mutual_information
                    self.mutual_information[j][i] = mutual_information

        scores = []

        for i in range(len(self.clusters)):
            if i == index:
                continue

            cluster = self.clusters[i]

            if len(cluster) > 0:
                scores.append(np.min([self.mutual_information[feature][c] for c in cluster]))

        if len(scores) > 0:
            return 1. - np.mean(scores)
        else:
            return mi.mutual_information_2d(X[:, feature], y, normalized=True)

import math
import numpy as np

from mutual_info import mutual_information, mutual_information_2d
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
    def __init__(self, base_clf, k, n, alpha=0.5):
        """
        :param base_clf: base classifier
        :param k: number of clusters
        :param n: number of elements in a single cluster
        :param alpha: score (inside cluster) coefficient
        :param 1 - alpha: diversification (between clusters) coefficient
        """
        self.alpha = alpha

        super(DeterministicSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        clusters = []

        for i in range(X.shape[1]):
            cluster = [i]
            discard = False

            while len(cluster) < self.n:
                best_score = -np.inf
                best_feature = -1

                for j in range(X.shape[1]):
                    if j in cluster:
                        continue

                    extended_cluster = cluster + [j]
                    current_score = self._score_inside_cluster(X, y, extended_cluster)

                    if current_score > best_score:
                        best_score = current_score
                        best_feature = j

                cluster.append(best_feature)

                for existing_cluster in clusters:
                    if set(cluster) == set(existing_cluster[:len(cluster)]):
                        discard = True

                        break

                if discard:
                    break

            if not discard:
                clusters.append(cluster)

        scores = []

        for cluster in clusters:
            scores.append(self._score_inside_cluster(X, y, cluster))

        selected_clusters = []

        while len(selected_clusters) < self.k:
            best_score = -np.inf
            best_cluster = []

            for cluster in clusters:
                if cluster in selected_clusters:
                    continue

                current_score = self._score_between_clusters(cluster, selected_clusters, scores[clusters.index(cluster)])

                if current_score > best_score:
                    best_score = current_score
                    best_cluster = cluster

            selected_clusters.append(best_cluster)

        self.clusters = selected_clusters
        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _score_inside_cluster(self, X, y, cluster, cv=5):
        return cross_validation.cross_val_score(self.base_clf, X[:, cluster], y, cv=cv).mean()

    def _score_between_clusters(self, cluster, selected_clusters, score_inside_cluster):
        hamming = 0

        for selected_cluster in selected_clusters:
            hamming += np.count_nonzero(cluster != selected_cluster)

        if len(selected_clusters) > 0:
            hamming /= len(selected_clusters) * self.k

        return self.alpha * score_inside_cluster - (1 - self.alpha) / (hamming + 1e-9)


class GaussianSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, b):
        """
        :param base_clf: base classifier
        :param k: number of clusters
        :param n: number of elements in a single cluster
        :param b: number of clusters computed before pruning
        """
        assert b >= k

        self.b = b

        super(GaussianSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        mi = [mutual_information_2d(X[:, i], y, normalized=True) for i in range(X.shape[1])]
        indices = np.argsort(mi)[::-1]
        clusters = [[indices[i]] for i in range(self.b)]

        for cluster in clusters:
            while len(cluster) < self.n:
                mi = []

                for i in range(X.shape[1]):
                    if i in cluster:
                        mi.append(np.inf)
                    else:
                        mi.append(mutual_information((X[:, cluster], np.expand_dims(X[:, i], 1)), k=5))

                cluster.append(np.argmin(mi))

        if self.k == self.b:
            selected_clusters = clusters
        else:
            selected_clusters = []

            for _ in range(self.k):
                scores = []
                counts, skf = self._correct_prediction_count(X, y, selected_clusters)

                for cluster in clusters:
                    if cluster in selected_clusters:
                        scores.append(-np.inf)
                    else:
                        scores.append(self._cluster_score(X, y, cluster, selected_clusters, counts, skf))

                selected_clusters.append(clusters[np.argmax(scores)])

        self.clusters = selected_clusters
        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _correct_prediction_count(self, X, y, selected_clusters, cv=5):
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

    def _cluster_score(self, X, y, cluster, selected_clusters, counts, skf):
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
            score += math.erf((count / float(len(selected_clusters) + 1) - threshold) * 1.96 * 3)

        return score


class QuickSubspaceClassifier(BaseSubspaceClassifier):
    def __init__(self, base_clf, k, n, alpha=0.5):
        """
        :param base_clf: base classifier
        :param k: number of clusters
        :param n: number of elements in a single cluster
        :param alpha: score (inside cluster) coefficient
        :param 1 - alpha: diversification (between clusters) coefficient
        """
        self.alpha = alpha

        super(QuickSubspaceClassifier, self).__init__(base_clf, k, n)

    def fit(self, X, y):
        self.clusters = [[] for _ in range(self.k)]

        for _ in range(self.n):
            for index in range(self.k):
                for feature in range(X.shape[1]):
                    cluster = self.clusters[index]
                    scores = [self.alpha * self._inside_score(X, y, cluster, feature) +
                              (1. - self.alpha) * self._outside_score(cluster, feature, index)]
                    cluster.append(np.argmax(scores))

        self.clfs = []

        for cluster in self.clusters:
            clf = clone(self.base_clf)
            clf.fit(X[:, cluster], y)
            self.clfs.append(clf)

        return self

    def _inside_score(self, X, y, cluster, feature, cv=5):
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

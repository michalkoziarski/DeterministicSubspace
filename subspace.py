import math
import numpy as np
import mutual_info as mi

from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def normalize(array):
    array = np.array(array)
    
    if np.nanmax(array) != np.nanmin(array):
        return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    else:
        return 0.5 * np.ones(len(array))


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
    def __init__(self, base_clf, k, n, alpha=0.5, b=1, omega=1.):
        """
        :param alpha: quality coefficient
        :param 1 - alpha: diversity coefficient
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
                quality_scores = []
                diversity_scores = []

                for feature in range(X.shape[1]):
                    if feature in cluster:
                        quality_scores.append(np.nan)
                        diversity_scores.append(np.nan)
                    else:
                        quality_scores.append(self._quality_score(X, y, feature))
                        diversity_scores.append(self._diversity_score(X, y, cluster, feature, index))

                quality_scores = normalize(quality_scores)
                diversity_scores = normalize(diversity_scores)
                scores = self.alpha * quality_scores + (1 - self.alpha) * diversity_scores
                cluster.append(np.nanargmax(scores))

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

    def _quality_score(self, X, y, feature):
        if not hasattr(self, 'scores'):
            self.scores = []

            for i in range(X.shape[1]):
                self.scores.append(cross_val_score(self.base_clf, X[:, i].reshape(-1, 1), y, cv=2).mean())

        return self.scores[feature]

    def _diversity_score(self, X, y, cluster, feature, index):
        cluster_count = 0
        similarities = []
        extended_cluster = cluster + [feature]

        for i in range(self.k):
            if i == index:
                continue

            if feature in self.clusters[i]:
                cluster_count += 1

            similarities.append(len(set(extended_cluster) & set(self.clusters[i])) / float(len(extended_cluster)))

        similarity_score = 1 - np.max(similarities)
        count_score = 1 - (float(cluster_count) / (self.k - 1))

        return (similarity_score + count_score) / 2

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
    def _quality_score(self, X, y, feature):
        if not hasattr(self, 'scores'):
            self.scores = []

            for i in range(X.shape[1]):
                self.scores.append(mi.mutual_information_2d(X[:, i], y))

        return self.scores[feature]


class CorrDeterministicSubspaceClassifier(DeterministicSubspaceClassifier):
    def _quality_score(self, X, y, feature):
        if not hasattr(self, 'scores'):
            self.scores = []

            for i in range(X.shape[1]):
                self.scores.append(np.abs(np.corrcoef(X[:, i], y)[0][1]))

        return self.scores[feature]

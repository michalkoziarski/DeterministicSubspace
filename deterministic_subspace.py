import numpy as np

from base_subspace import BaseSubspaceClassifier
from sklearn import cross_validation
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder


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

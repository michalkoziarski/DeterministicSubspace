import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


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

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity


class ParzenKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, h=1.0):
        self.h = h

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.priors = [float(len(X[y == label])) / len(X) for label in self.labels]
        self.observations = [X[y == label] for label in self.labels]

        return self

    def predict(self, X):
        predictions = []

        for sample in X:
            p = []

            for i in range(len(self.labels)):
                p.append(self.parzen_estimation(sample, i) * self.priors[i])

            predictions.append(self.labels[np.argmax(p)])

        return np.array(predictions)

    def parzen_estimation(self, sample, cls):
        k_n = 0.

        for obs in self.observations[cls]:
            k_n += self._window(self._kernel(obs, sample, self.h), self.h)

        return (k_n / len(self.observations[cls])) / (self.h ** self.observations[cls].shape[1])

    @staticmethod
    def _kernel(x, x_i, h):
        return (x - x_i) / h

    @staticmethod
    def _window(x, h):
        for v in x:
            if np.abs(v) > (h / 2.):
                return 0

        return 1


class NNKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='gaussian', bandwith=1.0):
        self.kernel = kernel
        self.bandwith = bandwith

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.kdes = [KernelDensity(kernel=self.kernel, bandwidth=self.bandwith).fit(X[y == label])
                     for label in self.labels]

        return self

    def predict(self, X):
        scores = []
        predictions = []

        for kde in self.kdes:
            scores.append(kde.score_samples(X))

        scores = np.array(scores)

        for i in range(len(X)):
            predictions.append(self.labels[np.argmax(scores[:, i])])

        return predictions


class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, covariance_type='diag', n_iter=100):
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.clf = None

    def fit(self, X, y):
        n_classes = len(np.unique(y))

        self.clf = GMM(n_components=n_classes,
                       covariance_type=self.covariance_type,
                       n_iter=self.n_iter,
                       init_params='wc')

        self.clf.means_ = np.array([X[y == i].mean(axis=0) for i in xrange(n_classes)])
        self.clf.fit(X, y)

        return self

    def predict(self, X):
        return self.clf.predict(X)

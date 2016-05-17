import numpy as np

from scipy.stats import kde
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity


class ParzenKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.labels = np.unique(y)
        self.priors = [float(len(X[y == label])) / len(X) for label in self.labels]
        self.kdes = []

        for label in self.labels:
            observations = X[y == label]
            observations += np.random.random(observations.shape) * 1e-5

            self.kdes.append(kde.gaussian_kde(observations.T, bw_method='scott'))

        return self

    def predict(self, X):
        predictions = []

        for sample in X:
            p = []

            for i in range(len(self.labels)):
                p.append(self.kdes[i].evaluate(sample) * self.priors[i])

            predictions.append(self.labels[np.argmax(p)])

        return np.array(predictions)


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

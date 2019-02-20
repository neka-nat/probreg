from __future__ import print_function
from __future__ import division
import numpy as np
from sklearn import mixture
from sklearn import svm


class GMM(object):
    def __init__(self, n_gmm_components=800):
        self._n_gmm_components = n_gmm_components

    def init(self):
        self._clf = mixture.GaussianMixture(n_components=self._n_gmm_components,
                                            covariance_type='spherical')

    def compute(self, data):
        self._clf.fit(data)
        return self._clf.means_, self._clf.weights_


class OneClassSVM(object):
    def __init__(self, ndim, sigma, gamma=0.5, nu=0.1):
        self._ndim = ndim
        self._sigma = sigma
        self._gamma = gamma
        self._nu = nu

    def init(self):
        self._clf = svm.OneClassSVM(nu=self._nu, kernel="rbf", gamma=self._gamma)

    def compute(self, data):
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma**2, self._ndim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z


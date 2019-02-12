from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
from scipy.optimize import minimize
from sklearn import mixture
from sklearn import svm
import open3d as o3
import transformations as trans
from . import transformation as tf
from . import gauss_transform as gt
from . import math_utils as mu


@six.add_metaclass(abc.ABCMeta)
class L2DistRegistration():
    def __init__(self, source, feature_gen,
                 sigma=1.0, delta=0.9, use_estimated_sigma=True):
        self._source = source
        self._feature_gen = feature_gen
        self._feature_gen.init()
        self._tf_type = tf.RigidTransformation
        self._sigma = sigma
        self._delta = delta
        self._use_estimated_sigma = use_estimated_sigma
        if not self._source is None and self._use_estimated_sigma:
            self._estimate_sigma(self._source)
        if not self._source is None:
            self._mu_source, self._phi_source = self._feature_gen.compute(self._source)

    def set_source(self, source):
        self._source = source
        if self._use_estimated_sigma:
            self._estimate_sigma(self._source)
        self._mu_source, self._phi_source = self._feature_gen.compute(self._source)

    def _estimate_sigma(self, data):
        ndata, ndim = data.shape
        data_hat = data - np.mean(data, axis=0)
        self._sigma = np.power(np.linalg.det(np.dot(data_hat.T, data_hat) / (ndata - 1)), 1.0 / (2.0 * ndim))

    def _to_transformation(self, theta):
        rot = trans.quaternion_matrix(theta[:4])[:3, :3]
        return self._tf_type(rot, theta[4:7])

    def obj_func(self, theta, *args):
        mu_source, phi_source, mu_target, phi_target, sigma = args
        z = np.power(2.0 * np.pi * sigma**2, mu_source.shape[1] * 0.5)
        tf_obj = self._to_transformation(theta)
        t_mu_source = tf_obj.transform(mu_source)
        gtrans = gt.GaussTransform(mu_target, np.sqrt(2.0) * sigma)
        phi_j_e = gtrans.compute(t_mu_source, phi_target / z)
        phi_mu_j_e = gtrans.compute(t_mu_source, phi_target * mu_target.T / z).T
        g = (phi_source * phi_j_e * t_mu_source.T - phi_source * phi_mu_j_e.T).T / (2.0 * sigma**2)
        d_rot = mu.diff_rot_from_quaternion(theta[:4])
        gtm0 = np.dot(g.T, mu_source)
        grad = np.concatenate([(gtm0 * d_rot).sum(axis=(1, 2)), g.sum(axis=0)])
        return -np.dot(phi_source, phi_j_e), grad

    def _optimization_cb(self, x):
        self._sigma *= self._delta

    def registration(self, target):
        mu_target, phi_target = self._feature_gen.compute(target)
        x0 = np.zeros(7)
        x0[0] = 1.0
        res = minimize(self.obj_func, x0,
                       args=(self._mu_source, self._phi_source,
                             mu_target, phi_target, self._sigma),
                       method='BFGS', jac=True,
                       callback=self._optimization_cb)
        return self._to_transformation(res.x)


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
    def __init__(self, ndim, sigma, gamma=0.5):
        self._ndim = ndim
        self._sigma = sigma
        self._gamma = gamma

    def init(self):
        self._clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=self._gamma)

    def compute(self, data):
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma**2, self._ndim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z


class GMMReg(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 n_gmm_components=800, use_estimated_sigma=True):
        super(GMMReg, self).__init__(source, GMM(n_gmm_components),
                                     sigma, delta,
                                     use_estimated_sigma)


class SupportVectorRegistration(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 gamma=0.5, use_estimated_sigma=True):
        super(SupportVectorRegistration, self).__init__(source,
                                                        OneClassSVM(source.shape[1],
                                                                    sigma, gamma),
                                                        sigma, delta,
                                                        use_estimated_sigma)

    def _estimate_sigma(self, data):
        super(SupportVectorRegistration, self)._estimate_sigma(data)
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))
        self._feature_gen.init()


def registration_gmmreg(source, target):
    gmmreg = GMMReg(np.asarray(source.points))
    return gmmreg.registration(np.asarray(target.points))

def registration_svr(source, target):
    svr = SupportVectorRegistration(np.asarray(source.points))
    return svr.registration(np.asarray(target.points))
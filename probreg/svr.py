from __future__ import print_function
from __future__ import division
from collections import namedtuple
import numpy as np
from scipy.optimize import minimize
from sklearn import svm
import open3d as o3
import transformations as trans
from . import transformation as tf
from . import gauss_transform as gt
from . import math_utils as mu


class SupportVectorRegistration():
    def __init__(self, source, gamma=0.5, sigma=1.0,
                 delta=0.9, use_estimated_sigma=True):
        self._source = source
        self._tf_type = tf.RigidTransformation
        self._gamma = gamma
        self._sigma = sigma
        self._delta = delta
        self._use_estimated_sigma = use_estimated_sigma
        if not self._source is None and self._use_estimated_sigma:
            self._sigma, self._gamma = self._estimate_sigma(self._source)
        self._clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=self._gamma)
        if not self._source is None:
            self._mu_source, self._phi_source = self._compute_svm(self._source)

    def set_source(self, source):
        self._source = source
        if self._use_estimated_sigma:
            self._sigma, self._gamma = self._estimate_sigma(self._source)
            self._clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=self._gamma)
        self._mu_source, self._phi_source = self._compute_svm(self._source)

    def _compute_svm(self, data):
        self._clf.fit(data)
        return self._clf.support_vectors_, self._clf.dual_coef_[0]

    def _estimate_sigma(self, data):
        ndata, ndim = data.shape
        data_hat = data - np.mean(data, axis=0)
        sigma = np.power(np.linalg.det(np.dot(data_hat.T, data_hat) / (ndata - 1)), 1.0 / (2.0 * ndim))
        return sigma, 1.0 / (2.0 * np.square(sigma))

    def _to_transformation(self, theta):
        rot = trans.quaternion_matrix(theta[:4])[:3, :3]
        return self._tf_type(rot, theta[4:7])

    def obj_func(self, theta, *args):
        mu_source, phi_source, mu_target, phi_target, sigma = args
        z = np.power(2.0 * np.pi * sigma**2, mu_source.shape[1] * 0.5)
        tf_obj = self._to_transformation(theta)
        t_mu_source = tf_obj.transform(mu_source)
        gtrans = gt.GaussTransform(mu_target, np.sqrt(2.0) * sigma)
        phi_j_e = gtrans.compute(t_mu_source, phi_target)
        phi_mu_j_e = gtrans.compute(t_mu_source, phi_target * mu_target.T).T
        g = (phi_source * phi_j_e * t_mu_source.T - phi_source * phi_mu_j_e.T).T / (2.0 * sigma**2)
        d_rot = mu.diff_rot_from_quaternion(theta[:4])
        gtm0 = np.dot(g.T, mu_source)
        grad = np.concatenate([(gtm0 * d_rot).sum(axis=(1, 2)), g.sum(axis=0)])
        return -np.dot(phi_source, phi_j_e) * z, grad * z

    def _optimization_cb(self, x):
        self._sigma *= self._delta

    def registration(self, target):
        mu_target, phi_target = self._compute_svm(target)
        x0 = np.zeros(7)
        x0[0] = 1.0
        res = minimize(self.obj_func, x0,
                       args=(self._mu_source, self._phi_source,
                             mu_target, phi_target, self._sigma),
                       method='BFGS', jac=True,
                       callback=self._optimization_cb)
        return self._to_transformation(res.x)


def registration_svr(source, target):
    svr = SupportVectorRegistration(np.asarray(source.points))
    return svr.registration(np.asarray(target.points))
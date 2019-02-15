from __future__ import print_function
from __future__ import division
import abc
import six
import numpy as np
import transformations as trans
from . import transformation as tf
from . import gauss_transform as gt
from . import se3_op as so


@six.add_metaclass(abc.ABCMeta)
class CostFunction():
    def __init__(self, tf_type):
        self._tf_type = tf_type

    @abc.abstractmethod
    def to_transformation(self, theta, *args):
        return None

    @abc.abstractmethod
    def initial(self, *args):
        return None

    @abc.abstractmethod
    def __call__(self, theta, *args):
        return None, None


def compute_l2_dist(mu_source, phi_source,
                    mu_target, phi_target, sigma):
    z = np.power(2.0 * np.pi * sigma**2, mu_source.shape[1] * 0.5)
    gtrans = gt.GaussTransform(mu_target, np.sqrt(2.0) * sigma)
    phi_j_e = gtrans.compute(mu_source, phi_target / z)
    phi_mu_j_e = gtrans.compute(mu_source, phi_target * mu_target.T / z).T
    g = (phi_source * phi_j_e * mu_source.T - phi_source * phi_mu_j_e.T).T / (2.0 * sigma**2)
    return -np.dot(phi_source, phi_j_e), g


class RigidCostFunction(CostFunction):
    def __init__(self):
        self._tf_type = tf.RigidTransformation

    def to_transformation(self, theta, *args):
        rot = trans.quaternion_matrix(theta[:4])[:3, :3]
        return self._tf_type(rot, theta[4:7])

    def initial(self, *args):
        x0 = np.zeros(7)
        x0[0] = 1.0
        return x0

    def __call__(self, theta, *args):
        mu_source, phi_source, mu_target, phi_target, sigma = args
        tf_obj = self.to_transformation(theta)
        t_mu_source = tf_obj.transform(mu_source)
        f, g = compute_l2_dist(t_mu_source, phi_source,
                               mu_target, phi_target, sigma)
        d_rot = so.diff_rot_from_quaternion(theta[:4])
        gtm0 = np.dot(g.T, mu_source)
        grad = np.concatenate([(gtm0 * d_rot).sum(axis=(1, 2)), g.sum(axis=0)])
        return f, grad


class TPSCostFunction(CostFunction):
    def __init__(self, ndim, alpha=1.0, beta=0.1):
        self._tf_type = tf.TPSTransformation
        self._ndim = ndim
        self._alpha = alpha
        self._beta = beta

    def to_transformation(self, theta, *args):
        mu_source = args[0]
        n_data = theta.shape[0] // self._ndim
        n_a = self._ndim * (self._ndim + 1)
        a = theta[:n_a].reshape(self._ndim + 1, self._ndim)
        v = theta[n_a:].reshape(n_data - self._ndim - 1, self._ndim)
        return self._tf_type(a, v, mu_source)

    def initial(self, *args):
        mu_source = args[0]
        a = np.r_[np.zeros((self._ndim, self._ndim)), np.ones((1, self._ndim))]
        v = np.zeros((mu_source.shape[0] - self._ndim - 1, self._ndim))
        return np.r_[a, v].flatten()

    def __call__(self, theta, *args):
        mu_source, phi_source, mu_target, phi_target, sigma = args
        tf_obj = self.to_transformation(theta, *args)
        basis, kernel = tf_obj.prepare(mu_source)
        t_mu_source = tf_obj.transform_basis(basis)
        bending = np.trace(np.dot(tf_obj.v.T, np.dot(kernel, tf_obj.v)))
        f1, g1 = compute_l2_dist(t_mu_source, phi_source,
                                 t_mu_source, phi_source, sigma)
        f2, g2 = compute_l2_dist(t_mu_source, phi_source,
                                 mu_target, phi_target, sigma)
        f = -f1 + 2.0 * f2
        g = -2.0 * g1 + 2.0 * g2
        grad = self._alpha * np.dot(basis.T, g)
        grad[self._ndim + 1:, :] += 2.0 * self._beta * np.dot(kernel, tf_obj.v)
        return self._alpha * f + self._beta * bending, grad.flatten()

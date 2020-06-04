from __future__ import print_function
from __future__ import division
import abc
import six
import numpy as np
import transforms3d as t3d
from . import transformation as tf
from . import gauss_transform as gt
from . import se3_op as so
from . import _ndt


@six.add_metaclass(abc.ABCMeta)
class CostFunction():
    def __init__(self, tf_type):
        self._tf_type = tf_type

    @abc.abstractmethod
    def to_transformation(self, theta):
        return None

    @abc.abstractmethod
    def initial(self):
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

    def to_transformation(self, theta):
        rot = t3d.quaternions.quat2mat(theta[:4])
        return self._tf_type(rot, theta[4:7])

    def initial(self):
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


class RigidCostFunctionWithCovariance(CostFunction):
    def __init__(self, d1=1.0, d2=0.05):
        self._tf_type = tf.RigidTransformation
        self._d1 = d1
        self._d2 = d2

    def to_transformation(self, theta):
        rot = t3d.euler.euler2mat(*theta[3:])
        return self._tf_type(rot, theta[:3])

    def initial(self):
        x0 = np.zeros(6)
        return x0

    def __call__(self, theta, *args):
        mu_source, sigma_source, mu_target, sigma_target, _ = args
        obj = _ndt.compute_objective_function(mu_source, sigma_source,
                                              mu_target, sigma_target,
                                              theta, self._d1, self._d2)
        return obj[0], obj[1], obj[2]


class TPSCostFunction(CostFunction):
    def __init__(self, control_pts,
                 alpha=1.0, beta=0.1):
        self._tf_type = tf.TPSTransformation
        self._alpha = alpha
        self._beta = beta
        self._control_pts = control_pts

    def to_transformation(self, theta):
        dim = self._control_pts.shape[1]
        n_data = theta.shape[0] // dim
        n_a = dim * (dim + 1)
        a = theta[:n_a].reshape(dim + 1, dim)
        v = theta[n_a:].reshape(n_data - dim - 1, dim)
        return self._tf_type(a, v, self._control_pts)

    def initial(self):
        dim = self._control_pts.shape[1]
        a = np.r_[np.zeros((1, dim)), np.identity(dim)]
        v = np.zeros((self._control_pts.shape[0] - dim - 1, dim))
        return np.r_[a, v].flatten()

    def __call__(self, theta, *args):
        dim = self._control_pts.shape[1]
        mu_source, phi_source, mu_target, phi_target, sigma = args
        tf_obj = self.to_transformation(theta)
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
        grad[dim + 1:, :] += 2.0 * self._beta * np.dot(kernel, tf_obj.v)
        return self._alpha * f + self._beta * bending, grad.flatten()

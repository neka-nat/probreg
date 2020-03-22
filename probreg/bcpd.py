from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
import scipy as sp
import open3d as o3
from . import transformation as tf
from . import math_utils as mu


EstepResult = namedtuple('EstepResult', ['pt1', 'p1', 'n_p', 'x_hat'])
MstepResult = namedtuple('MstepResult', ['transformation', 'u_hat', 'sigma_mat', 'alpha', 'sigma2', 'q'])


@six.add_metaclass(abc.ABCMeta)
class BayesianCoherentPointDrift():
    """Bayesian Coherent Point Drift algorithm.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
    """
    def __init__(self, source=None):
        self._source = source
        self._tf_type = None
        self._callbacks = []

    def set_source(self, source):
        self._source = source

    def set_callbacks(self, callbacks):
        self._callbacks.extend(callbacks)

    @abc.abstractmethod
    def _initialize(self, target):
        return MstepResult(None, None, None, None, None, None)

    def expectation_step(self, t_source, target, scale, alpha, sigma_mat, sigma2, w=0.0):
        """1st step for BCPD
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        dim = t_source.shape[1]
        pmat = np.stack([np.sum(np.square(target - ts), axis=1) for ts in t_source])
        pmat = np.exp(-pmat / (2.0 * sigma2))
        pmat /= (2.0 * np.pi * sigma2) ** (dim * 0.5)
        pmat = pmat.T
        pmat *= np.exp(-scale**2 / (2 * sigma2) * np.diag(sigma_mat) * dim)
        pmat *= (1.0 - w) * alpha
        den = w / target.shape[0] + (1.0 - w) * np.sum(pmat * alpha, axis=1)
        den[den==0] = np.finfo(np.float32).eps
        pmat = np.divide(pmat.T, den)

        pt1 = np.sum(pmat, axis=0)
        p1  = np.sum(pmat, axis=1)
        dnu_inv = 1.0 / np.kron(p1, np.ones(dim))
        px = np.dot(np.kron(pmat, np.identity(dim)), target)
        x_hat = np.multiply(px, dnu_inv)
        return EstepResult(pt1, p1, np.sum(p1), x_hat)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None):
        return None

    def registration(self, target, w=0.0,
                     maxiter=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target)
        q = res.q
        for _ in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(t_source, target, res.transformation.scale,
                                              res.alpha, res.sigma_mat, res.sigma2, w)
            res = self.maximization_step(target, res.transformation.rigid_trans, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            if abs(res.q - q) < tol:
                break
            q = res.q
        return res


class CombinedBCPD(BayesianCoherentPointDrift):
    def __init__(self, source=None, update_scale=True, update_nonrigid_term=True):
        super(CombinedBCPD, self).__init__(source)
        self._tf_type = tf.CombinedTransformation
        self._update_scale = update_scale
        self._update_nonrigid_term = update_nonrigid_term

    def _initialize(self, target):
        m, dim = self._source.shape
        self.gmat = mu.inverse_multiquadric_kernel(self._source, self._source)
        sigma2 = mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * ndim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(ndim), np.zeros(ndim)), None,
                            np.identity(m), 1.0 / m, sigma2, q)

    def maximization_step(self, target, rigid_trans, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, rigid_trans, estep_res,
                                       sigma2_p, self._update_scale)

    @staticmethod
    def _maximization_step(source, target, rigid_trans, estep_res,
                           sigma2_p=None, update_scale=True):
        pt1, p1, n_p, x_hat = estep_res
        dim = source.shape[1]
        m = source.shape[0]
        s2s2 = scale**2 / (sigma2_p**2)
        sigma_mat_inv = lmd * self.gmat + s2s2 * np.diag(p1)
        sigma_mat = np.linalg.inv(sigma_mat_inv)
        v_hat = s2s2 * np.matmul(np.matmul(np.kron(sigma_mat, np.identity(dim)),
                                           np.kron(p1, np.ones(dim))),
                                 rigid_trans.inverse().transform(target) - source)
        u_hat = source + v_hat
        alpha = np.exp(sp.special.psi(k + p1) - sp.special.psi(k * m + n_p))
        x_m = np.sum(p1 * x_hat, axis=0) / n_p
        sigma2_m = np.sum(p1 * np.diag(sigma_mat), axis=0) / n_p
        u_m = np.sum(p1 * u_hat, axis=0) / n_p
        u_hm = u_hat - u_m
        s_xu = np.sum(p1 * np.matmul(x_hat - x_m, u_hm.T), axis=0) / n_p
        s_uu = np.sum(p1 * np.matmul(u_hm, u_hm.T), axis=0) / n_p + sigma2_m * np.identity(dim)
        phi, s_xu_d, psih = np.linalg.svd(s_xu, full_matrices=True)
        c = np.ones(dim)
        c[-1] = np.linalg.det(np.dot(phi, psih))
        rot = np.matmul(phi * c, psih)
        tr_rsxu = np.trace(np.matmul(rot, s_xu))
        scale = tr_rxsu / np.trace(s_uu) if update_scale else 1.0
        t = x_m - scale * np.dot(rot, u_m)
        y_hat = rigid_trans.transform(source + v_hat)
        px = np.dot(np.kron(pmat, np.identity(dim)), target)
        sigma2 = (np.dot(target, pt1 * target) - 2.0 * np.dot(px.T, y_hat) + np.dot(y_hat, p1 * y_hat)) / (n_p * dim) + scale**2 * sigma2_m
        return MstepResult(CombinedTransformation(rot, t, scale, v_hat), u_hat, sigma_mat, alpha, sigma2, q)


def registration_bcpd(source, target, w=0.0, maxiter=50, tol=0.001,
                      callbacks=[], **kargs):
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    bcpd = CombinedBCPD(cv(source), **kargs)
    bcpd.set_callbacks(callbacks)
    return bcpd.registration(cv(target),
                             w, maxiter, tol)
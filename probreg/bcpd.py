from __future__ import division, print_function

import abc
from collections import namedtuple
from typing import Any, Callable, List, Union

import numpy as np
import open3d as o3
import scipy.special as spsp
import six
from scipy.spatial import cKDTree

from . import math_utils as mu
from . import transformation as tf
from .log import log

EstepResult = namedtuple("EstepResult", ["nu_d", "nu", "n_p", "px", "x_hat"])
MstepResult = namedtuple("MstepResult", ["transformation", "u_hat", "sigma_mat", "alpha", "sigma2"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (tf.Transformation): Transformation from source to target.
        u_hat (numpy.ndarray): A parameter used in next Estep.
        sigma_mat (numpy.ndarray): A parameter used in next Estep.
        alpha (float): A parameter used in next Estep.
        sigma2 (float): Variance of Gaussian distribution.
"""


@six.add_metaclass(abc.ABCMeta)
class BayesianCoherentPointDrift:
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
        return MstepResult(None, None, None, None, None)

    def expectation_step(self, t_source, target, scale, alpha, sigma_mat, sigma2, w=0.0):
        """Expectation step for BCPD"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        dim = t_source.shape[1]
        pmat = np.stack([np.sum(np.square(target - ts), axis=1) for ts in t_source])
        pmat = np.exp(-pmat / (2.0 * sigma2))
        pmat /= (2.0 * np.pi * sigma2) ** (dim * 0.5)
        pmat = pmat.T
        pmat *= np.exp(-(scale ** 2) / (2 * sigma2) * np.diag(sigma_mat) * dim)
        pmat *= (1.0 - w) * alpha
        den = w / target.shape[0] + np.sum(pmat, axis=1)
        den[den == 0] = np.finfo(np.float32).eps
        pmat = np.divide(pmat.T, den)

        nu_d = np.sum(pmat, axis=0)
        nu = np.sum(pmat, axis=1)
        nu_inv = 1.0 / np.kron(nu, np.ones(dim))
        px = np.dot(np.kron(pmat, np.identity(dim)), target.ravel())
        x_hat = np.multiply(px, nu_inv).reshape(-1, dim)
        return EstepResult(nu_d, nu, np.sum(nu), px.reshape(-1, dim), x_hat)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None):
        return None

    def registration(self, target, w=0.0, maxiter=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target)
        target_tree = cKDTree(target, leafsize=10)
        rmse = None
        for i in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(
                t_source, target, res.transformation.rigid_trans.scale, res.alpha, res.sigma_mat, res.sigma2, w
            )
            res = self.maximization_step(target, res.transformation.rigid_trans, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            tmp_rmse = mu.compute_rmse(t_source, target_tree)
            log.debug("Iteration: {}, Criteria: {}".format(i, tmp_rmse))
            if not rmse is None and abs(rmse - tmp_rmse) < tol:
                break
            rmse = tmp_rmse
        return res.transformation


class CombinedBCPD(BayesianCoherentPointDrift):
    def __init__(self, source=None, lmd=2.0, k=1.0e20, gamma=1.0):
        super(CombinedBCPD, self).__init__(source)
        self._tf_type = tf.CombinedTransformation
        self.lmd = lmd
        self.k = k
        self.gamma = gamma

    def _initialize(self, target):
        m, dim = self._source.shape
        self.gmat = mu.inverse_multiquadric_kernel(self._source, self._source)
        self.gmat_inv = np.linalg.inv(self.gmat)
        sigma2 = self.gamma * mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(dim), np.zeros(dim)), None, np.identity(m), 1.0 / m, sigma2)

    def maximization_step(self, target, rigid_trans, estep_res, sigma2_p=None):
        return self._maximization_step(
            self._source, target, rigid_trans, estep_res, self.gmat_inv, self.lmd, self.k, sigma2_p
        )

    @staticmethod
    def _maximization_step(source, target, rigid_trans, estep_res, gmat_inv, lmd, k, sigma2_p=None):
        nu_d, nu, n_p, px, x_hat = estep_res
        dim = source.shape[1]
        m = source.shape[0]
        s2s2 = rigid_trans.scale ** 2 / (sigma2_p ** 2)
        sigma_mat_inv = lmd * gmat_inv + s2s2 * np.diag(nu)
        sigma_mat = np.linalg.inv(sigma_mat_inv)
        residual = rigid_trans.inverse().transform(x_hat) - source
        v_hat = s2s2 * np.matmul(
            np.multiply(np.kron(sigma_mat, np.identity(dim)), np.kron(nu, np.ones(dim))), residual.ravel()
        ).reshape(-1, dim)
        u_hat = source + v_hat
        alpha = np.exp(spsp.psi(k + nu) - spsp.psi(k * m + n_p))
        x_m = np.sum(nu * x_hat.T, axis=1) / n_p
        sigma2_m = np.sum(nu * np.diag(sigma_mat), axis=0) / n_p
        u_m = np.sum(nu * u_hat.T, axis=1) / n_p
        u_hm = u_hat - u_m
        s_xu = np.matmul(np.multiply(nu, (x_hat - x_m).T), u_hm) / n_p
        s_uu = np.matmul(np.multiply(nu, u_hm.T), u_hm) / n_p + sigma2_m * np.identity(dim)
        phi, _, psih = np.linalg.svd(s_xu, full_matrices=True)
        c = np.ones(dim)
        c[-1] = np.linalg.det(np.dot(phi, psih))
        rot = np.matmul(phi * c, psih)
        tr_rsxu = np.trace(np.matmul(rot, s_xu))
        scale = tr_rsxu / np.trace(s_uu)
        t = x_m - scale * np.dot(rot, u_m)
        y_hat = rigid_trans.transform(source + v_hat)
        s1 = np.dot(target.ravel(), np.kron(nu_d, np.ones(dim)) * target.ravel())
        s2 = np.dot(px.ravel(), y_hat.ravel())
        s3 = np.dot(y_hat.ravel(), np.kron(nu, np.ones(dim)) * y_hat.ravel())
        sigma2 = (s1 - 2.0 * s2 + s3) / (n_p * dim) + scale ** 2 * sigma2_m
        return MstepResult(tf.CombinedTransformation(rot, t, scale, v_hat), u_hat, sigma_mat, alpha, sigma2)


def registration_bcpd(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    **kwargs: Any,
) -> tf.Transformation:
    """BCPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional) : Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Returns:
        probreg.Transformation: Estimated transformation.
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    bcpd = CombinedBCPD(cv(source), **kwargs)
    bcpd.set_callbacks(callbacks)
    return bcpd.registration(cv(target), w, maxiter)

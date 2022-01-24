from __future__ import division, print_function
from typing import Any, Callable, List, Union

import abc
from collections import namedtuple

import numpy as np
import open3d as o3
import six

from . import math_utils as mu
from . import transformation as tf
from .log import log

EstepResult = namedtuple("EstepResult", ["pt1", "p1", "px", "n_p"])
MstepResult = namedtuple("MstepResult", ["transformation", "sigma2", "q"])


@six.add_metaclass(abc.ABCMeta)
class CoherentPointDrift:
    """Coherent Point Drift algorithm.
    This is an abstract class.
    Based on this class, it is inherited by rigid, affine, nonrigid classes
    according to the type of transformation.
    In this class, Estimation step in EM algorithm is implemented and
    Maximazation step is implemented in the inherited classes.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(self, source=None, use_cuda=False):
        self._source = source
        self._tf_type = None
        self._callbacks = []
        if use_cuda:
            import cupy as cp
            from . import cupy_utils

            self.xp = cp
            self.cupy_utils = cupy_utils
            self._squared_kernel_sum = cupy_utils.squared_kernel_sum
        else:
            self.xp = np
            self._squared_kernel_sum = mu.squared_kernel_sum

    def set_source(self, source):
        self._source = source

    def set_callbacks(self, callbacks):
        self._callbacks.extend(callbacks)

    @abc.abstractmethod
    def _initialize(self, target):
        return MstepResult(None, None, None)

    def expectation_step(self, t_source, target, sigma2, w=0.0):
        """Expectation step for CPD"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        pmat = self.xp.stack([self.xp.sum(self.xp.square(target - ts), axis=1) for ts in t_source])
        pmat = self.xp.exp(-pmat / (2.0 * sigma2))

        c = (2.0 * np.pi * sigma2) ** (t_source.shape[1] * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        den = self.xp.sum(pmat, axis=0)
        den[den == 0] = self.xp.finfo(np.float32).eps
        den += c

        pmat = self.xp.divide(pmat, den)
        pt1 = self.xp.sum(pmat, axis=0)
        p1 = self.xp.sum(pmat, axis=1)
        px = self.xp.dot(pmat, target)
        return EstepResult(pt1, p1, px, np.sum(p1))

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p, xp=self.xp)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None, xp=np):
        return None

    def registration(self, target, w=0.0, maxiter=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target)
        q = res.q
        for i in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(target, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            log.debug("Iteration: {}, Criteria: {}".format(i, res.q))
            if abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidCPD(CoherentPointDrift):
    """Coherent Point Drift for rigid transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        update_scale (bool, optional): If this flag is True, compute the scale parameter.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(self, source=None, update_scale=True, tf_init_params={}, use_cuda=False):
        super(RigidCPD, self).__init__(source, use_cuda)
        self._tf_type = tf.RigidTransformation
        self._update_scale = update_scale
        self._tf_init_params = tf_init_params

    def _initialize(self, target):
        dim = self._source.shape[1]
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        if len(self._tf_init_params) == 0:
            self._tf_init_params = {"rot": self.xp.identity(dim), "t": self.xp.zeros(dim)}
        if not "xp" in self._tf_init_params:
            self._tf_init_params["xp"] = self.xp
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p, self._update_scale, self.xp)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None, update_scale=True, xp=np):
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        c = xp.ones(dim)
        c[-1] = xp.linalg.det(xp.dot(u, vh))
        rot = xp.dot(u * c, vh)
        tr_atr = np.trace(xp.dot(a.T, rot))
        tr_yp1y = np.trace(xp.dot(source_hat.T * p1, source_hat))
        scale = tr_atr / tr_yp1y if update_scale else 1.0
        t = mu_x - scale * xp.dot(rot, mu_y)
        tr_xp1x = xp.trace(xp.dot(target_hat.T * pt1, target_hat))
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x + tr_yp1y - scale * tr_atr) / (n_p * dim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale ** 2) * tr_yp1y) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.RigidTransformation(rot, t, scale, xp=xp), sigma2, q)


class AffineCPD(CoherentPointDrift):
    """Coherent Point Drift for affine transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(self, source=None, tf_init_params={}, use_cuda=False):
        super(AffineCPD, self).__init__(source, use_cuda)
        self._tf_type = tf.AffineTransformation
        self._tf_init_params = tf_init_params

    def _initialize(self, target):
        dim = self._source.shape[1]
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        if len(self._tf_init_params) == 0:
            self._tf_init_params = {"b": self.xp.identity(dim), "t": self.xp.zeros(dim)}
        if not "xp" in self._tf_init_params:
            self._tf_init_params["xp"] = self.xp
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None, xp=np):
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        yp1y = xp.dot(source_hat.T * p1, source_hat)
        b = xp.linalg.solve(yp1y.T, a.T).T
        t = mu_x - xp.dot(b, mu_y)
        tr_xp1x = xp.trace(xp.dot(target_hat.T * pt1, target_hat))
        tr_xpyb = xp.trace(xp.dot(a, b.T))
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        tr_ab = xp.trace(xp.dot(a, b.T))
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.AffineTransformation(b, t), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    """Coherent Point Drift for nonrigid transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(self, source=None, beta=2.0, lmd=2.0, use_cuda=False):
        super(NonRigidCPD, self).__init__(source, use_cuda)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self._tf_obj = None
        if not self._source is None:
            self._tf_obj = self._tf_type(None, self._source, self._beta, self.xp)

    def set_source(self, source):
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p, self._tf_obj, self._lmd, self.xp)

    def _initialize(self, target):
        dim = self._source.shape[1]
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        self._tf_obj.w = self.xp.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p, tf_obj, lmd, xp=np):
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        w = xp.linalg.solve((p1 * tf_obj.g).T + lmd * sigma2_p * xp.identity(source.shape[0]), px - (source.T * p1).T)
        t = source + xp.dot(tf_obj.g, w)
        tr_xp1x = xp.trace(xp.dot(target.T * pt1, target))
        tr_pxt = xp.trace(xp.dot(px.T, t))
        tr_tpt = xp.trace(xp.dot(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)

class ConstrainedNonRigidCPD(CoherentPointDrift):
    """
       Extended Coherent Point Drift for nonrigid transformation.
       Like CoherentPointDrift, but allows to add point correspondance constraints
       See: https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/ECPD2016.pdf

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        use_cuda (bool, optional): Use CUDA.
        idx_source (numpy.ndarray of ints, optional): Indices in source matrix
            for which a correspondance is known
        idx_target (numpy.ndarray of ints, optional): Indices in target matrix
            for which a correspondance is known
        alpha (float): Degree of reliability of priors. 
            Approximately between 1e-8 (highly reliable) and 1 (highly unreliable)
    """
    def __init__(self, source=None, beta=2.0, lmd=2.0, alpha=1e-8, use_cuda=False, 
                 idx_source=None, idx_target=None):
        super(ConstrainedNonRigidCPD, self).__init__(source, use_cuda)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self.alpha= alpha
        self._tf_obj = None
        self.idx_source, self.idx_target= idx_source, idx_target
        if not self._source is None:
            self._tf_obj = self._tf_type(None, self._source, self._beta, self.xp)

    def set_source(self, source):
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res,
                   sigma2_p, self._tf_obj, self._lmd,
                   self.alpha, self.p1_tilde, self.px_tilde, self.xp)

    def _initialize(self, target):
        dim = self._source.shape[1]
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        self._tf_obj.w = self.xp.zeros_like(self._source)
        self.p_tilde= self.xp.zeros((self._source.shape[0], target.shape[0]))
        if self.idx_source is not None and self.idx_target is not None:
            self.p_tilde[self.idx_source, self.idx_target]= 1
        self.p1_tilde = self.xp.sum(self.p_tilde, axis=1)
        self.px_tilde = self.xp.dot(self.p_tilde, target)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p, tf_obj, lmd, 
                           alpha, p1_tilde, px_tilde, xp=np):
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        w = xp.linalg.solve( (p1 * tf_obj.g).T + sigma2_p / alpha * (p1_tilde * tf_obj.g).T \
                                               + lmd * sigma2_p * xp.identity(source.shape[0]),
                             px - (source.T * p1).T + sigma2_p / alpha * ( px_tilde - (source.T * p1_tilde).T )
                            )
        t = source + xp.dot(tf_obj.g, w)
        tr_xp1x = xp.trace(xp.dot(target.T * pt1, target))
        tr_pxt = xp.trace(xp.dot(px.T, t))
        tr_tpt = xp.trace(xp.dot(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)

def registration_cpd(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    tf_type_name: str = "rigid",
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    use_cuda: bool = False,
    **kwargs: Any,
):
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid', 'nonrigid_constrained')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
        use_cuda (bool, optional): Use CUDA.

    Keyword Args:
        update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
            then the scale is treated. The default is true.
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).
    """
    xp = np
    if use_cuda:
        import cupy as cp

        xp = cp
    cv = lambda x: xp.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    if tf_type_name == "rigid":
        cpd = RigidCPD(cv(source), use_cuda=use_cuda, **kwargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(cv(source), use_cuda=use_cuda, **kwargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(cv(source), use_cuda=use_cuda, **kwargs)
    elif tf_type_name == 'nonrigid_constrained':
        cpd = ConstrainedNonRigidCPD(cv(source), use_cuda=use_cuda, **kwargs)
    else:
        raise ValueError("Unknown transformation type %s" % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(cv(target), w, maxiter, tol)

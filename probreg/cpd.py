from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
import open3d as o3
from . import transformation as tf
from . import gauss_transform as gt
from . import math_utils as mu


EstepResult = namedtuple('EstepResult', ['pt1', 'p1', 'px', 'n_p'])
MstepResult = namedtuple('MstepResult', ['transformation', 'sigma2', 'q'])


@six.add_metaclass(abc.ABCMeta)
class CoherentPointDrift():
    """Coherent Point Drift algorithm.
    This is an abstract class.
    Based on this class, it is inherited by rigid, affine, nonrigid classes
    according to the type of transformation.
    In this class, Estimation step in EM algorithm is implemented and
    Maximazation step is implemented in the inherited classes.

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
        return MstepResult(None, None, None)

    def expectation_step(self, t_source, target, sigma2, w=0.0):
        """Expectation step for CPD
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        ndim = t_source.shape[1]
        h = np.sqrt(2.0 * sigma2)
        c = (2.0 * np.pi * sigma2) ** (ndim * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        gtrans = gt.GaussTransform(t_source, h)
        kt1 = gtrans.compute(target)
        kt1[kt1==0] = np.finfo(np.float32).eps
        a = 1.0 / (kt1 + c)
        pt1 = 1.0 - c * a
        gtrans = gt.GaussTransform(target, h)
        p1 = gtrans.compute(t_source, a)
        px = gtrans.compute(t_source, np.tile(a, (ndim, 1)) * target.T).T
        return EstepResult(pt1, p1, px, np.sum(p1))

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
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(target, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            if abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidCPD(CoherentPointDrift):
    def __init__(self, source=None, update_scale=True):
        super(RigidCPD, self).__init__(source)
        self._tf_type = tf.RigidTransformation
        self._update_scale = update_scale

    def _initialize(self, target):
        ndim = self._source.shape[1]
        sigma2 = mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * ndim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(ndim), np.zeros(ndim)), sigma2, q)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res,
                                       sigma2_p, self._update_scale)

    @staticmethod
    def _maximization_step(source, target, estep_res,
                           sigma2_p=None, update_scale=True):
        pt1, p1, px, n_p = estep_res
        ndim = source.shape[1]
        mu_x = np.sum(px, axis=0) / n_p
        mu_y = np.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = np.dot(px.T, source_hat) - np.outer(mu_x, np.dot(p1.T, source_hat))
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        c = np.ones(ndim)
        c[-1] = np.linalg.det(np.dot(u, vh))
        rot = np.dot(u * c, vh)
        tr_atr = np.trace(np.dot(a.T, rot))
        tr_yp1y = np.trace(np.dot(source_hat.T * p1, source_hat))
        scale = tr_atr / tr_yp1y if update_scale else 1.0
        t = mu_x - scale * np.dot(rot, mu_y)
        tr_xp1x = np.trace(np.dot(target_hat.T * pt1, target_hat))
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * ndim)
        else:
            sigma2 = (tr_xp1x + tr_yp1y - scale * tr_atr) / (n_p * ndim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale ** 2) * tr_yp1y) / (2.0 * sigma2)
        q += ndim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.RigidTransformation(rot, t, scale), sigma2, q)


class AffineCPD(CoherentPointDrift):
    def __init__(self, source=None):
        super(AffineCPD, self).__init__(source)
        self._tf_type = tf.AffineTransformation

    def _initialize(self, target):
        ndim = self._source.shape[1]
        sigma2 = mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * ndim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(ndim), np.zeros(ndim)),
                           sigma2, q)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None):
        pt1, p1, px, n_p = estep_res
        ndim = source.shape[1]
        mu_x = np.sum(px, axis=0) / n_p
        mu_y = np.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = np.dot(px.T, source_hat) - np.outer(mu_x, np.dot(p1.T, source_hat))
        yp1y = np.dot(source_hat.T * p1, source_hat)
        b = np.linalg.solve(yp1y.T, a.T).T
        t = mu_x - np.dot(b, mu_y)
        tr_xp1x = np.trace(np.dot(target_hat.T * pt1, target_hat))
        tr_xpyb = np.trace(np.dot(a, b.T))
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * ndim)
        tr_ab = np.trace(np.dot(a, b.T))
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += ndim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.AffineTransformation(b, t), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    def __init__(self, source=None, beta=2.0, lmd=2.0):
        super(NonRigidCPD, self).__init__(source)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self._tf_obj = None
        if not self._source is None:
            self._tf_obj = self._tf_type(None, self._source, self._beta)

    def set_source(self, source):
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res,
                                       sigma2_p, self._tf_obj, self._lmd)

    def _initialize(self, target):
        ndim = self._source.shape[1]
        sigma2 = mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * ndim * 0.5 * np.log(sigma2)
        self._tf_obj.w = np.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(source, target, estep_res, sigma2_p, tf_obj, lmd):
        pt1, p1, px, n_p = estep_res
        ndim = source.shape[1]
        w = np.linalg.solve((p1 * tf_obj.g).T + lmd * sigma2_p * np.identity(source.shape[0]),
                            px - (source.T * p1).T)
        t = source + np.dot(tf_obj.g, w)
        tr_xp1x = np.trace(np.dot(target.T * pt1, target))
        tr_pxt = np.trace(np.dot(px.T, t))
        tr_tpt = np.trace(np.dot(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * ndim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


def registration_cpd(source, target, tf_type_name='rigid',
                     w=0.0, maxiter=50, tol=0.001,
                     callbacks=[], **kargs):
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.PointCloud) else x)
    if tf_type_name == 'rigid':
        cpd = RigidCPD(cv(source), **kargs)
    elif tf_type_name == 'affine':
        cpd = AffineCPD(cv(source), **kargs)
    elif tf_type_name == 'nonrigid':
        cpd = NonRigidCPD(cv(source), **kargs)
    else:
        raise ValueError('Unknown transformation type %s' % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(cv(target),
                            w, maxiter, tol)
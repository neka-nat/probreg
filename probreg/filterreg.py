from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
import open3d as o3
from . import transformation as tf
from . import gaussian_filtering as gf
from . import gauss_transform as gt
from . import se3_op as so
from . import _kabsch as kabsch
from . import _pt2pl as pt2pl
from . import math_utils as mu


EstepResult = namedtuple('EstepResult', ['m0', 'm1', 'm2', 'nx'])
MstepResult = namedtuple('MstepResult', ['transformation', 'sigma2', 'q'])

@six.add_metaclass(abc.ABCMeta)
class FilterReg():
    """FilterReg
    FilterReg is similar to CPD, and the speed performance is improved.
    In this algorithm, not only point-to-point alignment but also
    point-to-plane alignment are implemented.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        target_normals (numpy.ndarray, optional): Normals of target points.
        sigma2 (Float, optional): Variance parameter. If this variable is None,
            the variance is updated in Mstep.
    """
    def __init__(self, source=None, target_normals=None,
                 sigma2=None):
        self._source = source
        self._target_normals = target_normals
        self._sigma2 = sigma2
        self._update_sigma2 = self._sigma2 is None
        self._tf_type = None
        self._tf_result = None
        self._callbacks = []

    def set_source(self, source):
        self._source = source

    def set_target_normals(self, target_normals):
        self._target_normals = target_normals

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    def expectation_step(self, t_source, target, y, sigma2,
                         objective_type='pt2pt', alpha=0.015):
        """Expectation step
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        m, ndim = t_source.shape
        n = target.shape[0]
        sigma = np.sqrt(sigma2)
        fx = t_source / sigma
        fy = target / sigma
        zero_m1 = np.zeros((m, 1))
        zeros_md = np.zeros((m, y.shape[1]))
        fin = np.r_[fx, fy]
        ph = gf.Permutohedral(fin)
        if ph.get_lattice_size() < n * alpha:
            ph = gf.Permutohedral(fin, False)
        vin0 = np.r_[zero_m1, np.ones((n, 1))]
        vin1 = np.r_[zeros_md, y]
        m0 = ph.filter(vin0, m).flatten()[:m]
        m1 = ph.filter(vin1, m)[:m]
        if self._update_sigma2:
            vin2 = np.r_[zero_m1,
                         np.expand_dims(np.square(y).sum(axis=1), axis=1)]
            m2 = ph.filter(vin2, m).flatten()[:m]
        else:
            m2 = None
        if objective_type == 'pt2pt':
            nx = None
        elif objective_type == 'pt2pl':
            vin = np.r_[zeros_md, self._target_normals / dem]
            nx = ph.filter(vin, m)[:m]
        else:
            raise ValueError('Unknown objective_type: %s.' % objective_type)
        return EstepResult(m0, m1, m2, nx)

    def maximization_step(self, t_source, target, estep_res, w=0.0,
                          objective_type='pt2pt'):
        return self._maximization_step(t_source, target, estep_res,
                                       self._tf_result, self._sigma2, w,
                                       objective_type)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(t_source, target, estep_res, sigma2, w=0.0,
                           objective_type='pt2pt'):
        return None

    def registration(self, target, w=0.0,
                     objective_type='pt2pt',
                     maxiter=50, tol=0.001,
                     feature_fn=lambda x: x):
        assert not self._tf_type is None, "transformation type is None."
        q = None
        ftarget = feature_fn(target)
        if self._update_sigma2:
            self._sigma2 = mu.squared_kernel_sum(self._source, target)
        for _ in range(maxiter):
            t_source = self._tf_result.transform(self._source)
            fsource = feature_fn(t_source)
            estep_res = self.expectation_step(fsource, ftarget, target,
                                              self._sigma2, objective_type)
            res = self.maximization_step(t_source, target, estep_res, w=w,
                                         objective_type=objective_type)
            self._tf_result = res.transformation
            self._sigma2 = res.sigma2
            for c in self._callbacks:
                c(self._tf_result)
            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidFilterReg(FilterReg):
    def __init__(self, source=None, target_normals=None,
                 sigma2=None):
        super(RigidFilterReg, self).__init__(source, target_normals, sigma2)
        self._tf_type = tf.RigidTransformation
        self._tf_result = self._tf_type()

    @staticmethod
    def _maximization_step(t_source, target, estep_res, trans_p, sigma2, w=0.0,
                           objective_type='pt2pt', maxiter=10, tol=1.0e-4):
        m, ndim = t_source.shape
        n = target.shape[0]
        assert ndim == 3, "ndim must be 3."
        m0, m1, m2, nx = estep_res
        tw = np.zeros(ndim * 2)
        c = w / (1.0 - w) * n / m
        m0[m0==0] = np.finfo(np.float32).eps
        m1m0 = np.divide(m1.T, m0).T
        m0m0 = m0 / (m0 + c)
        drxdx = np.sqrt(m0m0 * 1.0 / sigma2)
        if objective_type == 'pt2pt':
            dr, dt = kabsch.kabsch(t_source, m1m0, drxdx)
            rx = np.multiply(drxdx, (t_source - m1m0).T).T.sum(axis=1)
            rot, t = np.dot(dr, trans_p.rot), np.dot(trans_p.t, dr.T) + dt
            q = np.dot(rx.T, rx).sum()
        elif objective_type == 'pt2pl':
            nxm0 = (nx.T / m0).T
            tw, q = pt2pl.compute_twist_for_pt2pl(t_source, m1m0, nxm0, drxdx)
            rot, t = so.twist_mul(tw, trans_p.rot, trans_p.t)
        else:
            raise ValueError('Unknown objective_type: %s.' % objective_type)

        if not m2 is None:
            sigma2 = ((m0 * np.square(t_source).sum(axis=1) - 2.0 * (t_source * m1).sum(axis=1) + m2) / (m0 + c)).sum()
            sigma2 /= m0m0.sum()
        return MstepResult(tf.RigidTransformation(rot, t), sigma2, q)


def registration_filterreg(source, target, target_normals=None,
                           sigma2=None, objective_type='pt2pt', maxiter=50,
                           tol=0.001, feature_fn=lambda x: x,
                           callbacks=[], **kargs):
    """FilterReg registration

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.PointCloud) else x)
    frg = RigidFilterReg(cv(source), cv(target_normals), sigma2, **kargs)
    frg.set_callbacks(callbacks)
    return frg.registration(cv(target), objective_type=objective_type, maxiter=maxiter,
                            tol=tol, feature_fn=feature_fn)
from __future__ import division, print_function

import abc
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union

import numpy as np
import open3d as o3
import six

from . import _kabsch as kabsch
from . import _pt2pl as pt2pl
from . import gauss_transform as gt
from . import gaussian_filtering as gf
from . import math_utils as mu
from . import se3_op as so
from . import transformation as tf
from .log import log

try:
    from dq3d import dualquat, quat

    _imp_dq = True
except:
    _imp_dq = False

EstepResult = namedtuple("EstepResult", ["m0", "m1", "m2", "nx"])
MstepResult = namedtuple("MstepResult", ["transformation", "sigma2", "q"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (tf.Transformation): Transformation from source to target.
        sigma2 (float): Variance of Gaussian distribution.
        q (float): Result of likelihood.
"""


def dualquat_from_twist(tw):
    ang = np.linalg.norm(tw[:3])
    if ang < np.finfo(np.float32).eps:
        return dualquat(quat.identity(), tw[3:])
    return dualquat(quat(ang, tw[:3] / ang), tw[3:])


@six.add_metaclass(abc.ABCMeta)
class FilterReg:
    """FilterReg
    FilterReg is similar to CPD, and the speed performance is improved.
    In this algorithm, not only point-to-point alignment but also
    point-to-plane alignment are implemented.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        target_normals (numpy.ndarray, optional): Normals of target points.
        sigma2 (Float, optional): Variance parameter. If this variable is None,
            the variance is updated in Mstep.
        update_sigma2 (bool, optional): If this variable is True, Update sigma2 in the registration iteration.
    """

    def __init__(self, source=None, target_normals=None, sigma2=None, update_sigma2=False):
        self._source = source
        self._target_normals = target_normals
        self._sigma2 = sigma2
        self._update_sigma2 = update_sigma2
        self._tf_type = None
        self._tf_result = None
        self._callbacks = []

    def set_source(self, source):
        self._source = source

    def set_target_normals(self, target_normals):
        self._target_normals = target_normals

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    def expectation_step(self, t_source, target, y, sigma2, update_sigma2, objective_type="pt2pt", alpha=0.015):
        """Expectation step"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        m, _ = t_source.shape
        n = target.shape[0]
        sigma = np.sqrt(sigma2)
        fx = t_source / sigma
        fy = target / sigma
        zero_m1 = np.zeros((m, 1))
        zeros_md = np.zeros((m, y.shape[1]))
        fin = np.r_[fx, fy]
        ph = gf.Permutohedral(fin)
        if ph.get_lattice_size() > n * alpha:
            ph = gf.Permutohedral(fin, False)
        vin0 = np.r_[zero_m1, np.ones((n, 1))]
        vin1 = np.r_[zeros_md, y]
        m0 = ph.filter(vin0, m).flatten()[:m]
        m1 = ph.filter(vin1, m)[:m]
        if update_sigma2:
            vin2 = np.r_[zero_m1, np.expand_dims(np.square(y).sum(axis=1), axis=1)]
            m2 = ph.filter(vin2, m).flatten()[:m]
        else:
            m2 = None
        if objective_type == "pt2pt":
            nx = None
        elif objective_type == "pt2pl":
            vin = np.r_[zeros_md, self._target_normals]
            nx = ph.filter(vin, m)[:m]
        else:
            raise ValueError("Unknown objective_type: %s." % objective_type)
        return EstepResult(m0, m1, m2, nx)

    def maximization_step(self, t_source, target, estep_res, w=0.0, objective_type="pt2pt"):
        return self._maximization_step(
            t_source, target, estep_res, self._tf_result, self._sigma2, w, objective_type=objective_type
        )

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(t_source, target, estep_res, trans_p, sigma2, w=0.0, objective_type="pt2pt"):
        return None

    def registration(
        self, target, w=0.0, objective_type="pt2pt", maxiter=50, tol=0.001, min_sigma2=1.0e-4, feature_fn=lambda x: x
    ):
        assert not self._tf_type is None, "transformation type is None."
        q = None
        ftarget = feature_fn(target)
        if self._sigma2 is None:
            fsource = feature_fn(self._source)
            self._sigma2 = max(mu.squared_kernel_sum(fsource, ftarget), min_sigma2)
        for i in range(maxiter):
            t_source = self._tf_result.transform(self._source)
            fsource = feature_fn(t_source)
            estep_res = self.expectation_step(
                fsource, ftarget, target, self._sigma2, self._update_sigma2, objective_type
            )
            res = self.maximization_step(t_source, target, estep_res, w=w, objective_type=objective_type)
            if res.q is None:
                res = res._replace(q=q)
                break
            self._tf_result = res.transformation
            self._sigma2 = max(res.sigma2, min_sigma2)
            for c in self._callbacks:
                c(self._tf_result)
            log.debug("Iteration: {}, Criteria: {}".format(i, res.q))
            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidFilterReg(FilterReg):
    def __init__(self, source=None, target_normals=None, sigma2=None, update_sigma2=False, tf_init_params={}):
        super(RigidFilterReg, self).__init__(
            source=source, target_normals=target_normals, sigma2=sigma2, update_sigma2=update_sigma2
        )
        self._tf_type = tf.RigidTransformation
        self._tf_result = self._tf_type(**tf_init_params)

    @staticmethod
    def _maximization_step(t_source, target, estep_res, trans_p, sigma2, w=0.0, objective_type="pt2pt"):
        m, dim = t_source.shape
        n = target.shape[0]
        assert dim == 2 or dim == 3, "dim must be 2 or 3."
        m0, m1, m2, nx = estep_res
        tw = np.zeros(dim * 2)
        c = w / (1.0 - w) * n / m * (2.0 * sigma2 * np.pi) ** (dim / 2.0)
        nonzero_idx = m0 != 0
        if not nonzero_idx.any():
            return MstepResult(trans_p, sigma2, None)
        m0 = m0[nonzero_idx]
        m1 = m1[nonzero_idx]
        t_source_e = t_source[nonzero_idx]
        m1m0 = np.divide(m1.T, m0).T
        m0m0 = m0 / (m0 + c)
        drxdx = np.sqrt(m0m0 * 1.0 / sigma2)
        if objective_type == "pt2pt":
            if dim == 2:
                dr, dt = kabsch.kabsch2d(t_source_e, m1m0, drxdx)
            else:
                dr, dt = kabsch.kabsch(t_source_e, m1m0, drxdx)
            rx = np.multiply(drxdx, (t_source_e - m1m0).T).T
            rot, t = np.dot(dr, trans_p.rot), np.dot(trans_p.t, dr.T) + dt
            q = np.linalg.norm(rx, ord=2, axis=1).sum()
        elif objective_type == "pt2pl":
            nxm0 = (nx[nonzero_idx].T / m0).T
            tw, q = pt2pl.compute_twist_for_pt2pl(t_source_e, m1m0, nxm0, drxdx)
            rot, t = so.twist_mul(tw, trans_p.rot, trans_p.t)
        else:
            raise ValueError("Unknown objective_type: %s." % objective_type)

        if not m2 is None:
            m2 = m2[nonzero_idx]
            sigma2 = (
                (m0 * np.square(t_source_e).sum(axis=1) - 2.0 * (t_source_e * m1).sum(axis=1) + m2) / (m0 + c)
            ).sum()
            sigma2 /= 3.0 * m0m0.sum()
        return MstepResult(tf.RigidTransformation(rot, t), sigma2, q)


class DeformableKinematicFilterReg(FilterReg):
    def __init__(self, source=None, skinning_weight=None, sigma2=None):
        if not _imp_dq:
            raise RuntimeError("No dq3d python package, filterreg deformation model not available.")
        super(DeformableKinematicFilterReg, self).__init__(source, sigma2=sigma2)
        self._tf_type = tf.DeformableKinematicModel
        self._skinning_weight = skinning_weight
        self._tf_result = self._tf_type(
            [dualquat.identity() for _ in range(self._skinning_weight.n_nodes)], self._skinning_weight
        )

    @staticmethod
    def _maximization_step(
        t_source, target, estep_res, trans_p, sigma2, w=0.0, objective_type="", maxiter=50, tol=1.0e-4
    ):
        m, dim = t_source.shape
        n6d = dim * 2
        idx_6d = lambda i: slice(i * n6d, (i + 1) * n6d)
        n = target.shape[0]
        n_nodes = trans_p.weights.n_nodes
        assert dim == 3, "dim must be 3."
        m0, m1, m2, _ = estep_res
        tw = np.zeros(n_nodes * dim * 2)
        c = w / (1.0 - w) * n / m
        m0[m0 == 0] = np.finfo(np.float32).eps
        m1m0 = np.divide(m1.T, m0).T
        m0m0 = m0 / (m0 + c)
        drxdx = np.sqrt(m0m0 * 1.0 / sigma2)
        dxdz = np.apply_along_axis(so.diff_x_from_twist, 1, t_source)
        a = np.zeros((n_nodes * n6d, n_nodes * n6d))
        for pair in trans_p.weights.pairs_set():
            jtj_tw = np.zeros([n6d, n6d])
            for idx in trans_p.weights.in_pair(pair):
                drxdz = drxdx[idx] * dxdz[idx]
                w = trans_p.weights[idx]["val"]
                jtj_tw += w[0] * w[1] * np.dot(drxdz.T, drxdz)
            a[idx_6d(pair[0]), idx_6d(pair[1])] += jtj_tw
            a[idx_6d(pair[1]), idx_6d(pair[0])] += jtj_tw
        for _ in range(maxiter):
            x = np.zeros_like(t_source)
            for pair in trans_p.weights.pairs_set():
                for idx in trans_p.weights.in_pair(pair):
                    w = trans_p.weights[idx]["val"]
                    q0 = dualquat_from_twist(tw[idx_6d(pair[0])])
                    q1 = dualquat_from_twist(tw[idx_6d(pair[1])])
                    x[idx] = (w[0] * q0 + w[1] * q1).transform_point(t_source[idx])

            rx = np.multiply(drxdx, (x - m1m0).T).T
            b = np.zeros(n_nodes * n6d)
            for pair in trans_p.weights.pairs_set():
                j_tw = np.zeros(n6d)
                for idx in trans_p.weights.in_pair(pair):
                    drxdz = drxdx[idx] * dxdz[idx]
                    w = trans_p.weights[idx]["val"]
                    j_tw += w[0] * np.dot(drxdz.T, rx[idx])
                b[idx_6d(pair[0])] += j_tw

            dtw = np.linalg.lstsq(a, b, rcond=None)[0]
            tw -= dtw
            if np.linalg.norm(dtw) < tol:
                break

        dualquats = [dualquat_from_twist(tw[idx_6d(i)]) * dq for i, dq in enumerate(trans_p.dualquats)]
        if not m2 is None:
            sigma2 = ((m0 * np.square(t_source).sum(axis=1) - 2.0 * (t_source * m1).sum(axis=1) + m2) / (m0 + c)).sum()
            sigma2 /= 3.0 * m0m0.sum()
        q = np.dot(rx.T, rx).sum()
        return MstepResult(tf.DeformableKinematicModel(dualquats, trans_p.weights), sigma2, q)


def registration_filterreg(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    target_normals: Optional[np.ndarray] = None,
    sigma2: Optional[float] = None,
    update_sigma2: bool = False,
    w: float = 0,
    objective_type: str = "pt2pt",
    maxiter: int = 50,
    tol: float = 0.001,
    min_sigma2: float = 1.0e-4,
    feature_fn: Callable = lambda x: x,
    callbacks: List[Callable] = [],
    **kwargs: Any,
):
    """FilterReg registration

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
        sigma2 (float, optional): Variance of GMM. If `sigma2` is `None`, `sigma2` is automatically updated.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        min_sigma2 (float, optional): Minimum variance of GMM.
        feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Keyword Args:
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid).

    Returns:
        MstepResult: Result of the registration (transformation, sigma2, q)
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    frg = RigidFilterReg(cv(source), cv(target_normals), sigma2, update_sigma2, **kwargs)
    frg.set_callbacks(callbacks)
    return frg.registration(
        cv(target),
        w=w,
        objective_type=objective_type,
        maxiter=maxiter,
        tol=tol,
        min_sigma2=min_sigma2,
        feature_fn=feature_fn,
    )

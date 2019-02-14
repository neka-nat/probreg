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


EstepResult = namedtuple('EstepResult', ['m0', 'm1'])
MstepResult = namedtuple('MstepResult', ['transformation', 'sigma2', 'q'])

@six.add_metaclass(abc.ABCMeta)
class FilterReg():
    def __init__(self, source=None, sigma2=None):
        self._source = source
        self._sigma2 = sigma2
        self._tf_type = None
        self._tf_result = None

    def set_source(self, source):
        self._source = source

    def expectation_step(self, t_source, target, sigma2):
        """
        Expectation step
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        m, ndim = t_source.shape
        n = target.shape[0]
        sigma = np.sqrt(sigma2)
        fx = t_source / sigma
        fy = target / sigma
        dem = np.power(2.0 * np.pi * sigma2, ndim * 0.5)
        fin = np.r_[fx, fy]
        vin0 = np.r_[np.zeros((m, 1)), np.ones((n, 1)) / dem]
        vin1 = np.r_[np.zeros_like(fx), target / dem]
        m0 = gf.filter(fin, vin0).flatten()[:m]
        m1 = gf.filter(fin, vin1)[:m]
        return EstepResult(m0, m1)

    def maximization_step(self, t_source, target, estep_res, w=0.0):
        return self._maximization_step(t_source, target, estep_res,
                                       self._tf_result, self._sigma2, w)

    @abc.abstractstaticmethod
    def _maximization_step(t_source, target, estep_res, sigma2, w=0.0):
        return None

    def registration(self, target, w=0.0,
                     max_iteration=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        q = None
        for _ in range(max_iteration):
            t_source = self._tf_result.transform(self._source)
            estep_res = self.expectation_step(t_source, target, self._sigma2)
            res = self.maximization_step(t_source, target, estep_res, w=w)
            self._tf_result = res.transformation
            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidFilterReg(FilterReg):
    def __init__(self, source=None, sigma2=None):
        super(RigidFilterReg, self).__init__(source, sigma2)
        self._tf_type = tf.RigidTransformation
        self._tf_result = self._tf_type()

    @staticmethod
    def _maximization_step(t_source, target, estep_res, trans_p, sigma2, w=0.0,
                           max_iteration=10, tol=1.0e-4):
        m, ndim = t_source.shape
        n = target.shape[0]
        assert ndim == 3, "ndim must be 3."
        m0, m1 = estep_res
        tw = np.zeros(ndim * 2)
        c = w / (1.0 - w) * n / m
        m0[m0==0] = np.finfo(np.float32).eps
        m1m0 = m1 / np.tile(m0, (ndim, 1)).T
        drxdx = np.tile(np.sqrt(m0 / (m0 + c) * 1.0 / sigma2), (ndim, 1)).T
        for _ in range(max_iteration):
            x = tf.RigidTransformation(*so.twist_trans(tw)).transform(t_source)
            rx = drxdx * (x - m1m0)
            dxdz = np.apply_along_axis(lambda x: np.c_[so.skew(x).T, np.identity(ndim)],
                                       1, x)
            drxdth = np.einsum('ij,ijl->ijl', drxdx, dxdz)
            a = np.einsum('ijk,ijl->kl', drxdth, drxdth)
            b = np.einsum('ijk,ij->k', drxdth, rx)
            dtw = np.linalg.solve(a, b)
            tw -= dtw
            if np.linalg.norm(dtw) < tol:
                break
        rot, t = so.twist_mul(tw, trans_p.rot, trans_p.t)
        q = np.einsum('ij,ij', rx, rx)
        return MstepResult(tf.RigidTransformation(rot, t), sigma2, q)


def registration_filterreg(source, target, sigma2=None):
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.PointCloud) else x)
    frg = RigidFilterReg(cv(source), sigma2)
    return frg.registration(cv(target))
from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
from . import gauss_transform as gt

EstepResult = namedtuple('EstepResult', ['pt1', 'p1', 'px', 'n_p'])
RigidResult = namedtuple('RigidResult', ['rot', 't', 'scale', 'sigma2'])
AffineResult = namedtuple('AffineResult', ['affine', 't', 'sigma2'])
NonRigidResult = namedtuple('NonRigidResult', ['kernel', 'coeff', 'sigma2'])

def init_sigma2(source, target):
    n, d = target.shape
    m, _ = source.shape
    xx = np.reshape(target, (1, n, d))
    yy = np.reshape(source, (m, 1, d))
    xx = np.tile(xx, (m, 1, 1))
    yy = np.tile(yy, (1, n, 1))
    diff = xx - yy
    err  = np.multiply(diff, diff)
    return np.sum(err) / (d * m * n)

@six.add_metaclass(abc.ABCMeta)
class CoherentPointDrift():
    def __init__(self):
        self._gt = None

    @abc.abstractmethod
    def transform(self, points):
        return points

    def expectation_step(self, source, target, sigma2, w=0.0):
        assert source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        h = np.sqrt(2.0 * sigma2)
        c = (2.0 * np.pi * sigma2) ** (source.shape[1] * 0.5)
        c *= w / (1.0 - w) * source.shape[0] / target.shape[0]
        self._gt = gt.GaussTransform(target, h)
        kt1 = self._gt.compute(source)
        a = 1.0 / (kt1 + c)
        pt1 = 1 - c * a
        self._gt = gt.GaussTransform(source)
        p1 = self._gt.compute(target, a)
        px = self._gt.compute(target, a * source).T
        return EstepResult(pt1, p1, px, np.sum(p1))

    @abc.abstractclassmethod
    def maximization_step(self):
        return None

    @abc.abstractclassmethod
    def registration(self, source, target):
        return None

class RigidCPD(CoherentPointDrift):
    def __init__(self):
        super(RigidCPD, self).__init__()

    def transform(self, points, res=None):
        if res is None:
            return points
        rot, t, scale, _ = res
        return scale * np.dot(points, rot) + t

    def maximization_step(self, source, target, estep_res):
        pt1, p1, px, n_p = estep_res
        mu_x = np.sum(px, axis=0) / n_p
        mu_y = np.sum(np.dot(source, p1), axis=0) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = np.dot(target, px).T - n_p * np.dot(mu_x, mu_y)
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        c = np.identity(source.shape[1])
        c[-1, -1] = np.linalg.det(np.dot(u, vh.T))
        rot = np.dot(np.dot(u, c), v.T)
        tr_ar = np.trace(np.dot(a, rot))
        scale = tr_ar / np.trace(np.dot(np.dot(source_hat.T, np.diag(p1)), source_hat))
        t = mu_x - scale * np.dot(rot, mu_y)
        sigma2 = (np.trace(np.dot(np.dot(target_hat.T, np.diag(pt1)), target_hat)) - scale * tr_ar) / (n_p * source.shape[1])
        return RigidResult(rot, t, scale, sigma2)

    def registration(self, source, target, w=0.0, max_iteration=30):
        sigma2 = init_sigma2(source, target)
        res = RigidResult(np.identity(source.shape[1]), np.zeros(3), 1.0, sigma2)
        for _ in range(max_iteration):
            t_source = self.transform(source, res)
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(t_source, target, estep_res)
        return res

def registration_cpd(source, target, transform_type='rigid'):
    if transform_type == 'rigid':
        cpd = RigidCPD()
    else:
        raise ValueError('Unknown transform_type %s' % transform_type)
    return cpd.registration(source, target)
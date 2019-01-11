from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
from . import gauss_transform as gt

EstepResult = namedtuple('EstepResult', ['pt1', 'p1', 'px', 'n_p'])
RigidResult = namedtuple('RigidResult', ['rot', 't', 'scale', 'sigma2', 'q'])
AffineResult = namedtuple('AffineResult', ['affine', 't', 'sigma2', 'q'])
NonRigidResult = namedtuple('NonRigidResult', ['kernel', 'coeff', 'sigma2', 'q'])

def init_sigma2(source, target):
    m, d = source.shape
    n, _ = target.shape
    err = np.sum([np.sum(np.power(t - source, 2)) for t in target])
    return err / (d * m * n)

@six.add_metaclass(abc.ABCMeta)
class CoherentPointDrift():
    def __init__(self):
        self._gt = gt

    @staticmethod
    @abc.abstractmethod
    def transform(points, res):
        return points

    def expectation_step(self, source, target, sigma2, w=0.0):
        assert source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        h = np.sqrt(2.0 * sigma2)
        c = (2.0 * np.pi * sigma2) ** (source.shape[1] * 0.5)
        c *= w / (1.0 - w) * source.shape[0] / target.shape[0]
        fgt = self._gt.GaussTransform(target, h)
        kt1 = fgt.compute(source)
        a = 1.0 / (kt1 + c)
        pt1 = 1 - c * a
        fgt = self._gt.GaussTransform(source, h)
        p1 = fgt.compute(target, a)
        px = fgt.compute(target, a * source).T
        return EstepResult(pt1, p1, px, np.sum(p1))

    @abc.abstractclassmethod
    def maximization_step(self, source, target, estep_res):
        return None

    @abc.abstractclassmethod
    def registration(self, source, target,
                     w=0.0, max_iteration=50,
                     tolerance=0.001):
        return None

class RigidCPD(CoherentPointDrift):
    def __init__(self):
        super(RigidCPD, self).__init__()

    @staticmethod
    def transform(points, res):
        rot, t, scale, _, _ = res
        return scale * np.dot(points, rot) + t

    def maximization_step(self, source, target, estep_res):
        pt1, p1, px, n_p = estep_res
        d = source.shape[1]
        mu_x = np.sum(px, axis=0) / n_p
        mu_y = np.sum(np.dot(source, p1), axis=0) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = np.dot(target, px).T - n_p * np.dot(mu_x, mu_y)
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        c = np.identity(d)
        c[-1, -1] = np.linalg.det(np.dot(u, vh.T))
        rot = np.dot(np.dot(u, c), vh.T)
        tr_ar = np.trace(np.dot(a, rot))
        ypy = np.dot(np.dot(source_hat.T, np.diag(p1)), source_hat)
        scale = tr_ar / np.trace(ypy)
        t = mu_x - scale * np.dot(rot, mu_y)
        xpx = np.dot(np.dot(target_hat.T, np.diag(pt1)), target_hat)
        sigma2 = (np.trace(xpx) - scale * tr_ar) / (n_p * source.shape[1])
        q = (xpx - 2.0 * scale * tr_ar + (scale ** 2) * ypy) / (2.0 * sigma2) + d * n_p * 0.5 * np.log(sigma2)
        return RigidResult(rot, t, scale, sigma2, q)

    def registration(self, source, target,
                     w=0.0, max_iteration=50,
                     tolerance=0.001):
        d = source.shape[1]
        sigma2 = init_sigma2(source, target)
        q = -tolerance + 1.0- target.shape[0] * d * 0.5 * np.log(sigma2)
        res = RigidResult(np.identity(source.shape[1]), np.zeros(3), 1.0, sigma2, q)
        for _ in range(max_iteration):
            t_source = self.transform(source, res)
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(t_source, target, estep_res)
            if abs(res.q - q) < tolerance:
                break
            q = res.q
        return res

def registration_cpd(source, target, transform_type='rigid',
                     w=0.0, max_iteration=50, tolerance=0.001):
    if transform_type == 'rigid':
        cpd = RigidCPD()
    else:
        raise ValueError('Unknown transform_type %s' % transform_type)
    return cpd.registration(np.asarray(source.points), np.asarray(target.points),
                            w, max_iteration, tolerance)
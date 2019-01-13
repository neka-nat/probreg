from __future__ import print_function
from __future__ import division
import numpy as np
from . import _ifgt

def _gauss_transform_direct(source, target, weights, h):
    h2 = h * h
    fn = lambda t: np.dot(weights, np.exp(-np.sum(np.square(source - t), axis=1) / h2))
    return np.apply_along_axis(fn, 1, target)

class Direct(object):
    def __init__(self, source, h):
        self._source = source
        self._h = h

    def compute(self, target, weights):
        return _gauss_transform_direct(self._source, target, weights, self._h)

class GaussTransform(object):
    def __init__(self, source, h, method='ifgt', eps=1.0e-4):
        if method == 'direct':
            self._impl = Direct(source, h)
        elif method == 'ifgt':
            self._impl = _ifgt.Ifgt(source, h, eps)
        else:
            raise ValueError("unknown method type %s" % method)

    def compute(self, target, weights=None):
        if weights is None:
            weights = np.ones(target.shape[0])
        if weights.ndim == 1:
            return self._impl.compute(target, weights)
        elif weights.ndim == 2:
            return np.r_[[self._impl.compute(target, w) for w in weights]]
        else:
            raise ValueError("weights.ndim must be 1 or 2.")

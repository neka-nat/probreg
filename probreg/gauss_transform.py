from __future__ import print_function
from __future__ import division
import numpy as np
from . import _ifgt

def _gauss_transform_direct(source, target, weights, h):
    """
    \sum_{j} weights[j] * \exp{ - \frac{||target[i] - source[j]||^2}{h^2} }
    """
    h2 = h * h
    fn = lambda t: np.dot(weights, np.exp(-np.sum(np.square(t - source), axis=1) / h2))
    return np.apply_along_axis(fn, 1, target)

class Direct(object):
    def __init__(self, source, h):
        self._source = source
        self._h = h

    def compute(self, target, weights):
        return _gauss_transform_direct(self._source, target, weights, self._h)

class GaussTransform(object):
    def __init__(self, source, h, eps=1.0e-4, sw_h=0.3):
        if h < sw_h:
            self._impl = Direct(source, h)
        else:
            self._impl = _ifgt.Ifgt(source, h, eps)

    def compute(self, target, weights=None):
        if weights is None:
            weights = np.ones(target.shape[0])
        if weights.ndim == 1:
            return self._impl.compute(target, weights)
        elif weights.ndim == 2:
            return np.r_[[self._impl.compute(target, w) for w in weights]]
        else:
            raise ValueError("weights.ndim must be 1 or 2.")

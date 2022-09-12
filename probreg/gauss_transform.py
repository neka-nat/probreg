from __future__ import division, print_function

from typing import Optional

import numpy as np

from . import _ifgt


def _gauss_transform_direct(source: np.ndarray, target: np.ndarray, weights: np.ndarray, h: float) -> np.ndarray:
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

    def compute(self, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return _gauss_transform_direct(self._source, target, weights, self._h)


class GaussTransform(object):
    """Calculate Gauss Transform

    Args:
        source (numpy.ndarray): Source data.
        h (float): Bandwidth parameter of the Gaussian.
        eps (float): Small floating point used in Gauss Transform.
        sw_h (float): Value of the bandwidth parameter to
            switch between direct method and IFGT.
    """

    def __init__(self, source: np.ndarray, h: float, eps: float = 1.0e-4, sw_h: float = 0.01):
        self._m = source.shape[0]
        if h < sw_h:
            self._impl = Direct(source, h)
        else:
            self._impl = _ifgt.Ifgt(source, h, eps)

    def compute(self, target: np.ndarray, weights: Optional[np.ndarray] = None):
        """Compute gauss transform

        Args:
            target (numpy.ndarray): Target data.
            weights (numpy.ndarray): Weights of Gauss Transform.
        """
        if weights is None:
            weights = np.ones(self._m)
        if weights.ndim == 1:
            return self._impl.compute(target, weights)
        elif weights.ndim == 2:
            return np.r_[[self._impl.compute(target, w) for w in weights]]
        else:
            raise ValueError("weights.ndim must be 1 or 2.")

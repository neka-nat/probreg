from __future__ import division, print_function

import numpy as np
from scipy.spatial import cKDTree

from . import _math


class Normalizer(object):
    """Normalizer

    Args:
        scale (float, optional): Scale factor.
        centroid (numpy.array, optional): Central point.
    """

    def __init__(self, scale: float = 1.0, centroid: float = 0.0) -> None:
        self._scale = scale
        self._centroid = centroid

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._centroid) / self._scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self._scale + self._centroid


def squared_kernel_sum(x: np.ndarray, y: np.ndarray) -> float:
    return _math.squared_kernel(x, y).sum() / (x.shape[0] * x.shape[1] * y.shape[0])


def compute_rmse(source: np.ndarray, target_tree: cKDTree) -> float:
    return sum(target_tree.query(source)[0]) / source.shape[0]


def rbf_kernel(x: np.ndarray, y: np.ndarray, beta: float) -> float:
    return _math.rbf_kernel(x, y, beta)


def tps_kernel(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape[1] == y.shape[1], "x and y must have same dimensions."
    if x.shape[1] == 2:
        return _math.tps_kernel_2d(x, y)
    elif x.shape[1] == 3:
        return _math.tps_kernel_3d(x, y)
    else:
        raise ValueError("Invalid dimension of x: %d." % x.shape[1])


def inverse_multiquadric_kernel(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    return _math.inverse_multiquadric_kernel(x, y, c)

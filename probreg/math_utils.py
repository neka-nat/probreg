from __future__ import print_function
from __future__ import division
from . import _math


class Normalizer(object):
    def __init__(self, scale=1.0, centroid=0.0):
        self._scale = scale
        self._centroid = centroid

    def normalize(self, x):
        return (x - self._centroid) / self._scale

    def denormalize(self, x):
        return x * self._scale + self._centroid


def squared_kernel_sum(x, y):
    """
    """
    return _math.squared_kernel(x, y).sum() / (x.shape[0] * x.shape[1] * y.shape[0])


def rbf_kernel(x, y, beta):
    return _math.rbf_kernel(x, y, beta)


def tps_kernel(x, y):
    assert x.shape[1] == y.shape[1], "x and y must have same dimensions."
    if x.shape[1] == 2:
        return _math.tps_kernel_2d(x, y)
    elif x.shape[1] == 3:
        return _math.tps_kernel_3d(x, y)
    else:
        raise ValueError('Invalid dimension of x: %d.' % x.shape[1])

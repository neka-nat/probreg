from __future__ import division, print_function

import numpy as np

from . import _permutohedral_lattice


class Permutohedral(object):
    def __init__(self, p: np.ndarray, with_blur: bool = True):
        self._impl = _permutohedral_lattice.Permutohedral()
        self._impl.init(p.T, with_blur)

    def get_lattice_size(self):
        return self._impl.get_lattice_size()

    def filter(self, v: np.ndarray, start: int = 0):
        return self._impl.filter(v.T, start).T

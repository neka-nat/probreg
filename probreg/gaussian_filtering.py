from __future__ import print_function
from __future__ import division
import numpy as np
from . import _permutohedral_lattice


class Permutohedral(object):
    def __init__(self, p, with_blur=True):
        self._impl = _permutohedral_lattice.Permutohedral()
        self._impl.init(p.T, with_blur)

    def filter(self, v):
        return self._impl.filter(v.T).T

from __future__ import print_function
from __future__ import division
import numpy as np
from . import _permutohedral_lattice


def filter(p, v, with_blur=True):
    """Permutohedral lattice filter
    """
    return _permutohedral_lattice.filter(p.T, v.T, with_blur).T

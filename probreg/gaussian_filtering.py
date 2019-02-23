from __future__ import print_function
from __future__ import division
import numpy as np
from . import _permutohedral_lattice

def filter(p, v):
    """Permutohedral lattice filter
    """
    return _permutohedral_lattice.filter(p.T, v.T).T

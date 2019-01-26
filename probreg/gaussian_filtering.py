from __future__ import print_function
from __future__ import division
import numpy as np
from . import _permutohedral_lattice

def filter(p, v):
    return _permutohedral_lattice.filter(p, v)

from __future__ import print_function
from __future__ import division
import numpy as np
from . import _ifgt

class GaussTransform(object):
    def __init__(self, source, h, eps=1.0e-4):
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

if __name__ == "__main__":
    import numpy as np
    x = np.random.rand(5, 3)
    gt = GaussTransform(x, 1.0)
    print(gt.compute(x, np.random.rand(5)))
import unittest
import numpy as np
from probreg import gauss_transform as gt
from probreg import _ifgt

def gauss_transform_direct(source, target, weights, h):
    h2 = h * h
    g = np.zeros(target.shape[0])
    for j, t in enumerate(target):
        dist = np.sum(np.square(source - t), axis=1)
        g[j] = np.dot(weights, np.exp(-dist / h2))
    return g

class GaussTransformTest(unittest.TestCase):
    def test_k_center_clustering(self):
        k1 = np.array([0.0, 0.0])
        k2 = np.array([10.0, 10.0])
        n = 10
        k1s = k1 + np.random.rand(n, 2)
        k2s = k2 + np.random.rand(n, 2)
        x = np.r_[k1s, k2s]
        idxs = _ifgt._kcenter_clustering(x, 2)
        self.assertTrue((idxs[:n] != idxs[n:]).all())

    def test_gaussian_transform(self):
        x = np.random.rand(5, 3)
        y = np.random.rand(5, 3)
        w = np.random.rand(5)
        ans = gauss_transform_direct(x, y, w, 1.0)
        trans = gt.GaussTransform(x, 1.0)
        self.assertTrue(np.allclose(ans, trans.compute(y, w), rtol=1.0e-4))

if __name__ == "__main__":
    unittest.main()
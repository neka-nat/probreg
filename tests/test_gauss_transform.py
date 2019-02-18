import unittest
import numpy as np
from probreg import gauss_transform as gt
from probreg import _ifgt

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

    def test_gauss_transform(self):
        x = np.random.rand(10, 3)
        y = np.random.rand(5, 3)
        w = np.random.rand(10)
        h = 1.0
        ans = gt._gauss_transform_direct(x, y, w, h)
        trans = gt.GaussTransform(x, h, sw_h=0.0)
        self.assertTrue(np.allclose(ans, trans.compute(y, w), atol=1.0e-4, rtol=1.0e-4))
        h = 0.5
        ans = gt._gauss_transform_direct(x, y, w, h)
        trans = gt.GaussTransform(x, h, sw_h=0.0)
        self.assertTrue(np.allclose(ans, trans.compute(y, w), atol=1.0e-4, rtol=1.0e-4))

if __name__ == "__main__":
    unittest.main()
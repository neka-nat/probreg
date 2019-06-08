import unittest
import numpy as np
from probreg import gaussian_filtering as gf
from probreg import gauss_transform as gt

class GaussianFilteringTest(unittest.TestCase):
    def test_gaussian_filtering(self):
        x = np.random.rand(10, 1)
        v0 = np.r_[np.zeros((5, 1)), np.ones((5, 1))]
        v1 = np.r_[np.zeros((5, 1)), np.random.rand(5, 1)]
        ph = gf.Permutohedral(x)
        out0 = ph.filter(v0).flatten()[:5]
        out1 = ph.filter(v1).flatten()[:5]
        out2 = gt._gauss_transform_direct(x[5:, :], x[:5, :],
                                          v0.flatten()[5:], np.sqrt(2.0))
        out3 = gt._gauss_transform_direct(x[5:, :], x[:5, :],
                                          v1.flatten()[5:], np.sqrt(2.0))
        self.assertTrue(np.allclose((out0 / out1), (out2 / out3), atol=0, rtol=3.0e-1))

if __name__ == "__main__":
    unittest.main()
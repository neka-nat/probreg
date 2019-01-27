import unittest
import numpy as np
from probreg import gaussian_filtering as gf
from probreg import gauss_transform as gt

class GaussianFilteringTest(unittest.TestCase):
    def test_gaussian_filtering(self):
        x = np.random.rand(5, 3)
        y = np.random.rand(5, 3)
        v = np.r_[np.zeros((5, 1)), np.ones((5, 1))]
        out1 = gf.filter(np.r_[x, y], v).flatten()[:5]
        out2 = gt._gauss_transform_direct(y, x, np.ones(5), np.sqrt(2.0))
        print(out1, out2)
        self.assertTrue(np.allclose(out1, out2, atol=1.0e-4, rtol=1.0e-4))

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
from probreg import math_utils as mu

class MathUtilsTest(unittest.TestCase):
    def test_pairwise_squared_sum(self):
        n = 5
        dim = 3
        x = np.arange(n * dim).reshape((n, dim))
        ans = np.sum([np.sum((x[i] - x)**2) for i in range(n)]) / (n * n * dim)
        self.assertAlmostEqual(mu.squared_kernel_sum(x, x), ans)

    def test_rbf_kernel(self):
        x = np.random.rand(5, 3)
        g = mu.rbf_kernel(x, x, 1.0)
        self.assertTrue(np.allclose(g, g.T))

if __name__ == "__main__":
    unittest.main()
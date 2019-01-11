import unittest
import numpy as np
from probreg import math_utils as mu

class MathUtilsTest(unittest.TestCase):
    def test_mean_square_norm(self):
        n = 5
        dim = 3
        x = np.arange(n * dim).reshape((n, dim))
        ans = np.sum([np.sum((x[i] - x)**2) for i in range(n)]) / (n * n * dim)
        self.assertAlmostEquals(mu.mean_square_norm(x, x), ans)

if __name__ == "__main__":
    unittest.main()
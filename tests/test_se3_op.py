import unittest
import numpy as np
from probreg import se3_op as so

def diff_from_tw_orig(x):
    return np.array([[0.0, x[2], -x[1], 1.0, 0.0, 0.0],
                     [-x[2], 0.0, x[0], 0.0, 1.0, 0.0],
                     [x[1], -x[0], 0.0, 0.0, 0.0, 1.0]])

class Se3OpTest(unittest.TestCase):
    def test_diff_from_tw(self):
        x = np.random.rand(5, 3)
        dx_0 = so.diff_from_tw(x)
        dx_1 = np.apply_along_axis(diff_from_tw_orig, 1, x)
        self.assertTrue(np.allclose(dx_0, dx_1))

    def test_diff_from_tw2(self):
        x = np.random.rand(5, 3)
        dx = so.diff_from_tw(x)
        dx2_0 = so.diff_from_tw2(dx)
        dx = dx.reshape((-1, 6))
        dx2_1 = np.dot(dx.T, dx)
        self.assertTrue(np.allclose(dx2_0, dx2_1))

if __name__ == "__main__":
    unittest.main()
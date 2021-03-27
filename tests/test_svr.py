import unittest
import numpy as np
import transforms3d as t3d
import open3d as o3
from probreg import l2dist_regs
from probreg import transformation as tf


class SVRTest(unittest.TestCase):
    def setUp(self):
        pcd = o3.io.read_point_cloud('data/horse.ply')
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        self._source = np.asarray(pcd.points)
        rot = t3d.euler.euler2mat(*np.random.uniform(0.0, np.pi / 4, 3))
        self._tf = tf.RigidTransformation(rot, np.zeros(3))
        self._target = self._tf.transform(self._source)

    def test_svr_registration(self):
        res = l2dist_regs.registration_svr(self._source, self._target)
        self.assertTrue(np.allclose(t3d.euler.mat2euler(res.rot),
                                    t3d.euler.mat2euler(self._tf.rot), atol=1.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

if __name__ == "__main__":
    unittest.main()
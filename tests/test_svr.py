import unittest
import numpy as np
import transformations as trans
import open3d as o3
from probreg import l2dist_regs
from probreg import transformation as tf


class SVRTest(unittest.TestCase):
    def setUp(self):
        pcd = o3.io.read_point_cloud('data/horse.ply')
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        self._source = np.asarray(pcd.points)
        rot = trans.euler_matrix(*np.random.uniform(0.0, np.pi / 4, 3))
        self._tf = tf.RigidTransformation(rot[:3, :3], np.zeros(3))
        self._target = self._tf.transform(self._source)

    def test_svr_registration(self):
        res = l2dist_regs.registration_svr(self._source, self._target)
        res_rot = trans.identity_matrix()
        res_rot[:3, :3] = res.rot
        ref_rot = trans.identity_matrix()
        ref_rot[:3, :3] = self._tf.rot
        self.assertTrue(np.allclose(trans.euler_from_matrix(res_rot),
                                    trans.euler_from_matrix(ref_rot), atol=1.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

if __name__ == "__main__":
    unittest.main()
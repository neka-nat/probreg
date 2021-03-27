import unittest
import numpy as np
import transforms3d as t3d
import open3d as o3
from probreg import filterreg
from probreg import transformation as tf


def estimate_normals(pcd, params):
    pcd.estimate_normals(search_param=params)
    pcd.orient_normals_to_align_with_direction()


class FilterRegTest(unittest.TestCase):
    def setUp(self):
        pcd = o3.io.read_point_cloud('data/horse.ply')
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        estimate_normals(pcd, o3.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
        self._source = np.asarray(pcd.points)
        rot = t3d.euler.euler2mat(*np.random.uniform(0.0, np.pi / 4, 3))
        self._tf = tf.RigidTransformation(rot, np.zeros(3))
        self._target = self._tf.transform(self._source)
        self._target_normals = np.asarray(np.dot(pcd.normals, self._tf.rot.T))

    def test_filterreg_registration_pt2pt(self):
        res = filterreg.registration_filterreg(self._source, self._target)
        self.assertTrue(np.allclose(t3d.euler.mat2euler(res.transformation.rot),
                                    t3d.euler.mat2euler(self._tf.rot), atol=2.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.transformation.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

    @unittest.skip("Skip pt2pl test.")
    def test_filterreg_registration_pt2pl(self):
        res = filterreg.registration_filterreg(self._source, self._target, self._target_normals)
        self.assertTrue(np.allclose(t3d.euler.mat2euler(res.transformation.rot),
                                    t3d.euler.mat2euler(self._tf.rot), atol=2.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.transformation.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

if __name__ == "__main__":
    unittest.main()

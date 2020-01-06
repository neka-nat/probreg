import unittest
import numpy as np
import transformations as trans
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
        rot = trans.euler_matrix(*np.random.uniform(0.0, np.pi / 4, 3))
        self._tf = tf.RigidTransformation(rot[:3, :3], np.zeros(3))
        self._target = self._tf.transform(self._source)
        self._target_normals = np.asarray(np.dot(pcd.normals, self._tf.rot.T))

    def test_filterreg_registration_pt2pt(self):
        res = filterreg.registration_filterreg(self._source, self._target)
        res_rot = trans.identity_matrix()
        res_rot[:3, :3] = res.transformation.rot
        ref_rot = trans.identity_matrix()
        ref_rot[:3, :3] = self._tf.rot
        self.assertTrue(np.allclose(trans.euler_from_matrix(res_rot),
                                    trans.euler_from_matrix(ref_rot), atol=2.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.transformation.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

    @unittest.skip("Skip pt2pl test.")
    def test_filterreg_registration_pt2pl(self):
        res = filterreg.registration_filterreg(self._source, self._target, self._target_normals)
        res_rot = trans.identity_matrix()
        res_rot[:3, :3] = res.transformation.rot
        ref_rot = trans.identity_matrix()
        ref_rot[:3, :3] = self._tf.rot
        self.assertTrue(np.allclose(trans.euler_from_matrix(res_rot),
                                    trans.euler_from_matrix(ref_rot), atol=2.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(res.transformation.t, self._tf.t, atol=1.0e-2, rtol=1.0e-3))

if __name__ == "__main__":
    unittest.main()

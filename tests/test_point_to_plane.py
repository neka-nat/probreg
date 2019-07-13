import unittest
import numpy as np
import transformations as trans
import open3d as o3
from probreg import _pt2pl as pt2pl
from probreg import se3_op as so
from probreg import transformation as tf


class Point2PlaneTest(unittest.TestCase):
    def setUp(self):
        # Generate plane
        points = []
        normals = []
        resolution = 0.1
        for i in np.arange(0.0, 1.0, resolution):
            for j in np.arange(0.0, 1.0, resolution):
                 points.append(np.array([i, j, -0.5]))
                 normals.append(np.array([0.0, 0.0, 1.0]))
        pcd = o3.PointCloud()
        pcd.points = o3.Vector3dVector(np.array(points))
        pcd.normals = o3.Vector3dVector(np.array(normals))
        self._source = np.asarray(pcd.points)
        rot = trans.euler_matrix(np.rad2deg(1.0), 0.0, 0.0)
        self._tf = tf.RigidTransformation(rot[:3, :3], np.zeros(3))
        self._target = self._tf.transform(self._source)
        self._target_normals = np.asarray(np.dot(pcd.normals, self._tf.rot.T))

    def test_point_to_plane(self):
        tw = pt2pl.compute_twist_for_pt2pl(self._source.T, self._target.T, self._target_normals.T,
                                           np.ones(self._source.shape[0]))
        r, t = so.twist_mul(tw, np.identity(3), np.zeros(3))
        r0 = np.identity(4)
        r0[:3, :3] = r
        r1 = np.identity(4)
        r1[:3, :3] = self._tf.rot
        self.assertTrue(np.allclose(trans.euler_from_matrix(r0),
                                    trans.euler_from_matrix(r1), atol=5.0e-1, rtol=1.0e-1))
        self.assertTrue(np.allclose(t, self._tf.t, atol=5.0e-1, rtol=1.0e-3))

if __name__ == "__main__":
    unittest.main()
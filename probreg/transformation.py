import abc
import six
import numpy as np
import open3d as o3
from . import math_utils as mu


@six.add_metaclass(abc.ABCMeta)
class Transformation():
    def __init__(self):
        pass

    def transform(self, points,
                  array_type=o3.Vector3dVector):
        if isinstance(points, array_type):
            return array_type(self._transform(np.asarray(points)))
        return self._transform(points)

    @abc.abstractmethod
    def _transform(self, points):
        return points


class RigidTransformation(Transformation):
    def __init__(self, rot=np.identity(3),
                 t=np.zeros(3), scale=1.0):
        super(RigidTransformation, self).__init__()
        self.rot = rot
        self.t = t
        self.scale = scale

    def _transform(self, points):
        return self.scale * np.dot(points, self.rot.T) + self.t


class AffineTransformation(Transformation):
    def __init__(self, b=np.identity(3),
                 t=np.zeros(3)):
        super(AffineTransformation, self).__init__()
        self.b = b
        self.t = t

    def _transform(self, points):
        return np.dot(points, self.b.T) + self.t


class NonRigidTransformation(Transformation):
    def __init__(self, g, w):
        super(NonRigidTransformation, self).__init__()
        self.g = g
        self.w = w

    def _transform(self, points):
        return points + np.dot(self.g, self.w)


class TPSTransformation(Transformation):
    def __init__(self, a, v, control_pts=None):
        super(TPSTransformation, self).__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts

    def prepare(self, landmarks, control_pts=None):
        if control_pts is None:
            control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape
        pm = np.c_[np.ones((m, 1)), landmarks]
        pn = np.c_[np.ones((n, 1)), control_pts]
        u, _, _ = np.linalg.svd(pn)
        pp = u[:, d + 1:]
        kk = mu.tps_kernel(control_pts, control_pts)
        uu = mu.tps_kernel(landmarks, control_pts)
        basis = np.c_[pm, np.dot(uu, pp)]
        kernel = np.dot(pp.T, np.dot(kk, pp))
        return basis, kernel

    def transform_basis(self, basis):
        return np.dot(basis, np.r_[self.a, self.v])

    def _transform(self, points):
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)
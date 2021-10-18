import abc
import itertools

import numpy as np
import open3d as o3
import six

from . import math_utils as mu

try:
    from dq3d import op

    _imp_dq = True
except:
    _imp_dq = False


@six.add_metaclass(abc.ABCMeta)
class Transformation:
    def __init__(self, xp=np):
        self.xp = xp

    def transform(self, points, array_type=o3.utility.Vector3dVector):
        if isinstance(points, array_type):
            return array_type(self._transform(self.xp.asarray(points)))
        return self._transform(points)

    @abc.abstractmethod
    def _transform(self, points):
        return points


class RigidTransformation(Transformation):
    """Rigid Transformation

    Args:
        rot (numpy.ndarray, optional): Rotation matrix.
        t (numpy.ndarray, optional): Translation vector.
        scale (Float, optional): Scale factor.
        xp (module, optional): Numpy or Cupy.
    """

    def __init__(self, rot=np.identity(3), t=np.zeros(3), scale=1.0, xp=np):
        super(RigidTransformation, self).__init__(xp)
        self.rot = rot
        self.t = t
        self.scale = scale

    def _transform(self, points):
        return self.scale * self.xp.dot(points, self.rot.T) + self.t

    def inverse(self):
        return RigidTransformation(self.rot.T, -self.xp.dot(self.rot.T, self.t) / self.scale, 1.0 / self.scale)

    def __mul__(self, other):
        return RigidTransformation(
            self.xp.dot(self.rot, other.rot),
            self.t + self.scale * self.xp.dot(self.rot, other.t),
            self.scale * other.scale,
        )


class AffineTransformation(Transformation):
    """Affine Transformation

    Args:
        b (numpy.ndarray, optional): Affine matrix.
        t (numpy.ndarray, optional): Translation vector.
        xp (module, optional): Numpy or Cupy.
    """

    def __init__(self, b=np.identity(3), t=np.zeros(3), xp=np):
        super(AffineTransformation, self).__init__(xp)
        self.b = b
        self.t = t

    def _transform(self, points):
        return self.xp.dot(points, self.b.T) + self.t


class NonRigidTransformation(Transformation):
    """Nonrigid Transformation

    Args:
        w (numpy.array): Weights for kernel.
        points (numpy.array): Source point cloud data.
        beta (float, optional): Parameter for gaussian kernel.
        xp (module): Numpy or Cupy.
    """

    def __init__(self, w, points, beta=2.0, xp=np):
        super(NonRigidTransformation, self).__init__(xp)
        if xp == np:
            self.g = mu.rbf_kernel(points, points, beta)
        else:
            from . import cupy_utils

            self.g = cupy_utils.rbf_kernel(points, points, beta)
        self.w = w

    def _transform(self, points):
        return points + self.xp.dot(self.g, self.w)


class CombinedTransformation(Transformation):
    """Combined Transformation

    Args:
        rot (numpy.array, optional): Rotation matrix.
        t (numpy.array, optional): Translation vector.
        scale (float, optional): Scale factor.
        v (numpy.array, optional): Nonrigid term.
    """

    def __init__(self, rot=np.identity(3), t=np.zeros(3), scale=1.0, v=0.0):
        super(CombinedTransformation, self).__init__()
        self.rigid_trans = RigidTransformation(rot, t, scale)
        self.v = v

    def _transform(self, points):
        return self.rigid_trans._transform(points + self.v)


class TPSTransformation(Transformation):
    """Thin Plate Spline transformaion.

    Args:
        a (numpy.array): Affine matrix.
        v (numpy.array): Translation vector.
        control_pts (numpy.array): Control points.
        kernel (function, optional): Kernel function.
    """

    def __init__(self, a, v, control_pts, kernel=mu.tps_kernel):
        super(TPSTransformation, self).__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts
        self._kernel = kernel

    def prepare(self, landmarks):
        control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape
        pm = np.c_[np.ones((m, 1)), landmarks]
        pn = np.c_[np.ones((n, 1)), control_pts]
        u, _, _ = np.linalg.svd(pn)
        pp = u[:, d + 1 :]
        kk = self._kernel(control_pts, control_pts)
        uu = self._kernel(landmarks, control_pts)
        basis = np.c_[pm, np.dot(uu, pp)]
        kernel = np.dot(pp.T, np.dot(kk, pp))
        return basis, kernel

    def transform_basis(self, basis):
        return np.dot(basis, np.r_[self.a, self.v])

    def _transform(self, points):
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)


class DeformableKinematicModel(Transformation):
    """Deformable Kinematic Transformation

    Args:
        dualquats (:obj:`list` of :obj:`dq3d.dualquat`): Transformations for each link.
        weights (DeformableKinematicModel.SkinningWeight): Skinning weight.
    """

    class SkinningWeight(np.ndarray):
        """SkinningWeight
                Transformations and weights for each point.

        .       tf = SkinningWeight['val'][0] * dualquats[SkinningWeight['pair'][0]] + SkinningWeight['val'][1] * dualquats[SkinningWeight['pair'][1]]
        """

        def __new__(cls, n_points):
            return super(DeformableKinematicModel.SkinningWeight, cls).__new__(
                cls, n_points, dtype=[("pair", "i4", 2), ("val", "f4", 2)]
            )

        @property
        def n_nodes(self):
            return self["pair"].max() + 1

        def pairs_set(self):
            return itertools.permutations(range(self.n_nodes), 2)

        def in_pair(self, pair):
            """
            Return indices of the pairs equal to the given pair.
            """
            return np.argwhere((self["pair"] == pair).all(1)).flatten()

    @classmethod
    def make_weight(cls, pairs, vals):
        weights = cls.SkinningWeight(pairs.shape[0])
        weights["pair"] = pairs
        weights["val"] = vals
        return weights

    def __init__(self, dualquats, weights):
        if not _imp_dq:
            raise RuntimeError("No dq3d python package, deformable kinematic model not available.")
        super(DeformableKinematicModel, self).__init__()
        self.weights = weights
        self.dualquats = dualquats
        self.trans = [op.dlb(w[1], [self.dualquats[i] for i in w[0]]) for w in self.weights]

    def _transform(self, points):
        return np.array([t.transform_point(p) for t, p in zip(self.trans, points)])

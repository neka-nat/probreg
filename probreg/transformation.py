import abc
import six
import numpy as np
import open3d as o3


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
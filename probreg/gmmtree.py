from __future__ import print_function
from __future__ import division
from collections import namedtuple
import numpy as np
import open3d as o3
from . import _gmmtree
from . import transformation as tf
from . import se3_op as so

EstepResult = namedtuple('EstepResult', ['moments'])
MstepResult = namedtuple('MstepResult', ['transformation', 'q'])


class GMMTree():
    """GMM Tree

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        tree_level (int, optional): Maximum depth level of GMM tree.
        lambda_c (float, optional): Parameter that determine the pruning of GMM tree
    """
    def __init__(self, source=None, tree_level=2, lambda_c=0.01):
        self._source = source
        self._tree_level = tree_level
        self._lambda_c = lambda_c
        self._tf_type = tf.RigidTransformation
        self._tf_result = self._tf_type()
        self._callbacks = []
        if not self._source is None:
            self._nodes = _gmmtree.build_gmmtree(self._source,
                                                 self._tree_level,
                                                 0.001, 1.0e-4)

    def set_source(self, source):
        self._source = source
        self._nodes = _gmmtree.build_gmmtree(self._source,
                                             self._tree_level,
                                             0.001, 1.0e-4)

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    def expectation_step(self, target):
        res = _gmmtree.gmmtree_reg_estep(target, self._nodes,
                                         self._tree_level, self._lambda_c)
        return EstepResult(res)

    def maximization_step(self, estep_res, trans_p):
        moments = estep_res.moments
        n = len(moments)
        amat = np.zeros((n * 3, 6))
        bmat = np.zeros(n * 3)
        for i, m in enumerate(moments):
            if m[0] < np.finfo(np.float32).eps:
                continue
            lmd, nn = np.linalg.eigh(self._nodes[i][2])
            s = m[1] / m[0]
            nn = np.multiply(nn, np.sqrt(m[0] / lmd))
            sl = slice(3 * i, 3 * (i + 1))
            bmat[sl] = np.dot(nn.T, self._nodes[i][1]) - np.dot(nn.T, s)
            amat[sl, :3] = np.cross(s, nn.T)
            amat[sl, 3:] = nn.T
        x, q, _, _ = np.linalg.lstsq(amat, bmat, rcond=-1)
        rot, t = so.twist_mul(x, trans_p.rot, trans_p.t)
        return MstepResult(tf.RigidTransformation(rot, t), q)

    def registration(self, target, maxiter=20, tol=1.0e-4):
        q = None
        for _ in range(maxiter):
            t_target = self._tf_result.transform(target)
            estep_res = self.expectation_step(t_target)
            res = self.maximization_step(estep_res, self._tf_result)
            self._tf_result = res.transformation
            for c in self._callbacks:
                c(self._tf_result.inverse())
            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return MstepResult(self._tf_result.inverse(), res.q)


def registration_gmmtree(source, target, maxiter=20, tol=1.0e-4,
                         callbacks=[], **kargs):
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.PointCloud) else x)
    gt = GMMTree(cv(source), **kargs)
    gt.set_callbacks(callbacks)
    return gt.registration(cv(target), maxiter, tol)
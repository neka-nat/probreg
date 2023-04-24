from __future__ import division, print_function

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import open3d as o3

from . import _gmmtree
from . import se3_op as so
from . import transformation as tf
from .log import log

EstepResult = namedtuple("EstepResult", ["moments"])
MstepResult = namedtuple("MstepResult", ["transformation", "q"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (tf.Transformation): Transformation from source to target.
        q (float): Result of likelihood.
"""


class GMMTree:
    """GMM Tree

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        tree_level (int, optional): Maximum depth level of GMM tree.
        lambda_c (float, optional): Parameter that determine the pruning of GMM tree.
        lambda_s (float, optional): Parameter that tolerance for building GMM tree.
        tf_init_params (dict, optional): Parameters to initialize transformation.
    """

    def __init__(
        self,
        source: Optional[np.ndarray] = None,
        tree_level: int = 2,
        lambda_c: float = 0.01,
        lambda_s: float = 0.001,
        tf_init_params: Dict = {},
    ):
        self._source = source
        self._tree_level = tree_level
        self._lambda_c = lambda_c
        self._lambda_s = lambda_s
        self._tf_type = tf.RigidTransformation
        self._tf_result = self._tf_type(**tf_init_params)
        self._callbacks = []
        if not self._source is None:
            self._nodes = _gmmtree.build_gmmtree(self._source, self._tree_level, self._lambda_s, 1.0e-4)

    def set_source(self, source: np.ndarray) -> None:
        self._source = source
        self._nodes = _gmmtree.build_gmmtree(self._source, self._tree_level, self._lambda_s, 1.0e-4)

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    def expectation_step(self, target: np.ndarray) -> EstepResult:
        res = _gmmtree.gmmtree_reg_estep(target, self._nodes, self._tree_level, self._lambda_c)
        return EstepResult(res)

    def maximization_step(self, estep_res: EstepResult, trans_p: tf.Transformation) -> MstepResult:
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

    def registration(self, target: np.ndarray, maxiter: int = 20, tol: float = 1.0e-4) -> MstepResult:
        q = None
        for i in range(maxiter):
            t_target = self._tf_result.transform(target)
            estep_res = self.expectation_step(t_target)
            res = self.maximization_step(estep_res, self._tf_result)
            self._tf_result = res.transformation
            for c in self._callbacks:
                c(self._tf_result.inverse())
            log.debug("Iteration: {}, Criteria: {}".format(i, res.q))
            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return MstepResult(self._tf_result.inverse(), res.q)


def registration_gmmtree(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    maxiter: int = 20,
    tol: float = 1.0e-4,
    callbacks: List[Callable] = [],
    **kwargs: Any,
) -> MstepResult:
    """GMMTree registration

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Keyword Args:
        tree_level (int, optional): Maximum depth level of GMM tree.
        lambda_c (float, optional): Parameter that determine the pruning of GMM tree.
        lambda_s (float, optional): Parameter that tolerance for building GMM tree.
        tf_init_params (dict, optional): Parameters to initialize transformation.

    Returns:
        MstepResult: Result of the registration (transformation, q)
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    gt = GMMTree(cv(source), **kwargs)
    gt.set_callbacks(callbacks)
    return gt.registration(cv(target), maxiter, tol)

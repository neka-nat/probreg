from __future__ import division, print_function

import abc

import numpy as np
import open3d as o3
import six
from sklearn import mixture, svm


@six.add_metaclass(abc.ABCMeta)
class Feature:
    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def compute(self, data):
        return None

    def annealing(self):
        pass

    def __call__(self, data):
        return self.compute(data)


class FPFH(Feature):
    """Fast Point Feature Histograms

    Args:
        radius_normal (float): Radius search parameter for computing normal vectors
        radius_feature (float): Radius search parameter for computing FPFH.
    """

    def __init__(self, radius_normal: float = 0.1, radius_feature: float = 0.5):
        self._param_normal = o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        self._param_feature = o3.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)

    def init(self):
        pass

    def estimate_normals(self, pcd: o3.geometry.PointCloud):
        pcd.estimate_normals(search_param=self._param_normal)

    def compute(self, data: np.ndarray):
        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(data)
        self.estimate_normals(pcd)
        fpfh = o3.pipelines.registration.compute_fpfh_feature(pcd, self._param_feature)
        return fpfh.data.T


class GMM(Feature):
    """Feature points extraction using Gaussian mixture model

    Args:
        n_gmm_components (int): The number of mixture components.
    """

    def __init__(self, n_gmm_components: int = 800):
        self._n_gmm_components = n_gmm_components

    def init(self):
        self._clf = mixture.GaussianMixture(n_components=self._n_gmm_components, covariance_type="spherical")

    def compute(self, data: np.ndarray):
        self._clf.fit(data)
        return self._clf.means_, self._clf.weights_


class OneClassSVM(Feature):
    """Feature points extraction using One class SVM

    Args:
        dim (int): The dimension of samples.
        sigma (float): Veriance of the gaussian distribution made from parameters of SVM.
        gamma (float, optional): Coefficient for RBF kernel.
        nu (float, optional): An upper bound on the fraction of training errors
            and a lower bound of the fraction of support vectors.
        delta (float, optional): Anealing parameter for optimization.
    """

    def __init__(self, dim: int, sigma: float, gamma: float = 0.5, nu: float = 0.05, delta: float = 10.0):
        self._dim = dim
        self._sigma = sigma
        self._gamma = gamma
        self._nu = nu
        self._delta = delta

    def init(self):
        self._clf = svm.OneClassSVM(nu=self._nu, kernel="rbf", gamma=self._gamma)

    def compute(self, data: np.ndarray):
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma ** 2, self._dim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z

    def annealing(self):
        self._gamma *= self._delta

from __future__ import print_function
from __future__ import division
import abc
import six
import numpy as np
import open3d as o3
from sklearn import mixture
from sklearn import svm


@six.add_metaclass(abc.ABCMeta)
class Feature():
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
    def __init__(self, radius_normal=0.1, radius_feature=0.5):
        self._param_normal = o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        self._param_feature = o3.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)

    def init(self):
        pass

    def compute(self, data):
        pcd = o3.PointCloud()
        pcd.points = o3.Vector3dVector(data)
        o3.estimate_normals(pcd, self._param_normal)
        fpfh = o3.registration.compute_fpfh_feature(pcd, self._param_feature)
        return fpfh.data.T


class GMM(Feature):
    """Feature points extraction using Gaussian mixture model

    Args:
        n_gmm_components (int): The number of mixture components.
    """
    def __init__(self, n_gmm_components=800):
        self._n_gmm_components = n_gmm_components

    def init(self):
        self._clf = mixture.GaussianMixture(n_components=self._n_gmm_components,
                                            covariance_type='spherical')

    def compute(self, data):
        self._clf.fit(data)
        return self._clf.means_, self._clf.weights_


class OneClassSVM(Feature):
    """Feature points extraction using One class SVM

    Args:
        ndim (int): The dimension of samples.
        sigma (float): Veriance of the gaussian distribution made from parameters of SVM.
        gamma (float, optional): Coefficient for RBF kernel.
        nu (float, optional): An upper bound on the fraction of training errors
            and a lower bound of the fraction of support vectors.
        delta (float, optional): Anealing parameter for optimization.
    """
    def __init__(self, ndim, sigma, gamma=0.5, nu=0.05, delta=10.0):
        self._ndim = ndim
        self._sigma = sigma
        self._gamma = gamma
        self._nu = nu
        self._delta = delta

    def init(self):
        self._clf = svm.OneClassSVM(nu=self._nu, kernel="rbf", gamma=self._gamma)

    def compute(self, data):
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma**2, self._ndim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z

    def annealing(self):
        self._gamma *= self._delta
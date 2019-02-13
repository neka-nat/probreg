from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
from scipy.optimize import minimize
import open3d as o3
from . import features as ft
from . import cost_functions as cf


@six.add_metaclass(abc.ABCMeta)
class L2DistRegistration():
    def __init__(self, source, feature_gen, cost_fn,
                 sigma=1.0, delta=0.9,
                 use_estimated_sigma=True):
        self._source = source
        self._feature_gen = feature_gen
        self._feature_gen.init()
        self._cost_fn = cost_fn
        self._sigma = sigma
        self._delta = delta
        self._use_estimated_sigma = use_estimated_sigma
        if not self._source is None and self._use_estimated_sigma:
            self._estimate_sigma(self._source)
        if not self._source is None:
            self._mu_source, self._phi_source = self._feature_gen.compute(self._source)

    def set_source(self, source):
        self._source = source
        if self._use_estimated_sigma:
            self._estimate_sigma(self._source)
        self._mu_source, self._phi_source = self._feature_gen.compute(self._source)

    def _estimate_sigma(self, data):
        ndata, ndim = data.shape
        data_hat = data - np.mean(data, axis=0)
        self._sigma = np.power(np.linalg.det(np.dot(data_hat.T, data_hat) / (ndata - 1)), 1.0 / (2.0 * ndim))

    def _obj_func(self, theta, *args):
        mu_source, phi_source, mu_target, phi_target, sigma = args
        return self._cost_fn(theta, mu_source, phi_source,
                             mu_target, phi_target, sigma)

    def _optimization_cb(self, x):
        self._sigma *= self._delta

    def registration(self, target):
        mu_target, phi_target = self._feature_gen.compute(target)
        args = (self._mu_source, self._phi_source,
                mu_target, phi_target, self._sigma)
        res = minimize(self._obj_func,
                       self._cost_fn.initial(*args),
                       args=args,
                       method='BFGS', jac=True,
                       callback=self._optimization_cb)
        return self._cost_fn.to_transformation(res.x)


class RigidGMMReg(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 n_gmm_components=800, use_estimated_sigma=True):
        super(RigidGMMReg, self).__init__(source, ft.GMM(n_gmm_components),
                                          cf.RigidCostFunction(),
                                          sigma, delta,
                                          use_estimated_sigma)


class TPSGMMReg(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 n_gmm_components=800, use_estimated_sigma=True):
        super(TPSGMMReg, self).__init__(source, ft.GMM(n_gmm_components),
                                        cf.TPSCostFunction(source.shape[1]),
                                        sigma, delta,
                                        use_estimated_sigma)


class RigidSupportVectorRegistration(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 gamma=0.5, use_estimated_sigma=True):
        super(RigidSupportVectorRegistration, self).__init__(source,
                                                             ft.OneClassSVM(source.shape[1],
                                                                            sigma, gamma),
                                                             cf.RigidCostFunction(),
                                                             sigma, delta,
                                                             use_estimated_sigma)

    def _estimate_sigma(self, data):
        super(RigidSupportVectorRegistration, self)._estimate_sigma(data)
        self._feature_gen._sigma = self._sigma
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))
        self._feature_gen.init()


class TPSSupportVectorRegistration(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 gamma=0.5, use_estimated_sigma=True):
        super(TPSSupportVectorRegistration, self).__init__(source,
                                                           ft.OneClassSVM(source.shape[1],
                                                                          sigma, gamma),
                                                           cf.TPSCostFunction(source.shape[1]),
                                                           sigma, delta,
                                                           use_estimated_sigma)

    def _estimate_sigma(self, data):
        super(TPSSupportVectorRegistration, self)._estimate_sigma(data)
        self._feature_gen._sigma = self._sigma
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))
        self._feature_gen.init()


def registration_gmmreg(source, target, tf_type_name='rigid'):
    if tf_type_name == 'rigid':
        gmmreg = RigidGMMReg(np.asarray(source.points))
    elif tf_type_name == 'nonrigid':
        gmmreg = TPSGMMReg(np.asarray(source.points))
    else:
        raise ValueError('Unknown transform type %s' % tf_type_name)
    return gmmreg.registration(np.asarray(target.points))

def registration_svr(source, target, tf_type_name='rigid'):
    if tf_type_name == 'rigid':
        svr = RigidSupportVectorRegistration(np.asarray(source.points))
    elif tf_type_name == 'norigid':
        svr = TPSSupportVectorRegistration(np.asarray(source.points))
    else:
        raise ValueError('Unknown transform type %s' % tf_type_name)
    return svr.registration(np.asarray(target.points))
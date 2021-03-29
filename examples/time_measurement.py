from timeit import default_timer as timer
import numpy as np
import open3d as o3
import utils
from probreg import cpd
from probreg import l2dist_regs
from probreg import gmmtree
from probreg import filterreg

threshold = 0.001
max_iteration = 100

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd',  n_random=0,
                                                          orientation=np.deg2rad([0.0, 0.0, 10.0]))

start = timer()
res = o3.pipelines.registration.registration_icp(source, target, 0.5,
                                                 np.identity(4), o3.pipelines.registration.TransformationEstimationPointToPoint(),
                                                 o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
end = timer()
print('ICP(Open3D): ', end - start)

start = timer()
res = cpd.registration_cpd(source, target, maxiter=max_iteration, tol=threshold)
end = timer()
print('CPD: ', end - start)

start = timer()
res = l2dist_regs.registration_svr(source, target, opt_maxiter=max_iteration, opt_tol=threshold)
end = timer()
print('SVR: ', end - start)

start = timer()
res = gmmtree.registration_gmmtree(source, target, maxiter=max_iteration, tol=threshold)
end = timer()
print('GMMTree: ', end - start)

start = timer()
res = filterreg.registration_filterreg(source, target,
                                       sigma2=None, maxiter=max_iteration, tol=threshold)
end = timer()
print('FilterReg: ', end - start)
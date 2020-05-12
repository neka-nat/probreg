import numpy as np
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import open3d as o3
import transformations as trans
from probreg import cpd
from probreg import callbacks
import utils
import time

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd', voxel_size=0.003)
source = cp.asarray(source.points, dtype=cp.float32)
target = cp.asarray(target.points, dtype=cp.float32)

rcpd = cpd.RigidCPD(source, use_cuda=use_cuda)
start = time.time()
tf_param, _, _ = rcpd.registration(target)
elapsed = time.time() - start
print("time: ", elapsed)

rot = trans.identity_matrix()
rot[:3, :3] = to_cpu(tf_param.rot)
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.scale, to_cpu(tf_param.t))
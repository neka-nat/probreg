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
import transforms3d as trans
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

print("result: ", np.rad2deg(t3d.euler.mat2euler(to_cpu(tf_param.rot))),
      tf_param.scale, to_cpu(tf_param.t))

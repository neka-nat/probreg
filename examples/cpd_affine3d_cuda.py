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
from probreg import cpd
from probreg import callbacks
import utils
import time

source, target = utils.prepare_source_and_target_nonrigid_3d('face-x.txt', 'face-y.txt', voxel_size=3.0)
source = cp.asarray(source.points, dtype=cp.float32)
target = cp.asarray(target.points, dtype=cp.float32)

acpd = cpd.AffineCPD(source, use_cuda=use_cuda)
start = time.time()
tf_param, _, _ = acpd.registration(target)
elapsed = time.time() - start
print("time: ", elapsed)

print("result: ", to_cpu(tf_param.b), to_cpu(tf_param.t))
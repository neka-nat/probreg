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

source, target = utils.prepare_source_and_target_nonrigid_3d('face-x.txt', 'face-y.txt', voxel_size=5.0)
source_pt = cp.asarray(source.points, dtype=cp.float32)
target_pt = cp.asarray(target.points, dtype=cp.float32)

acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
start = time.time()
tf_param, _, _ = acpd.registration(target_pt)
elapsed = time.time() - start
print("time: ", elapsed)

print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))

result = tf_param.transform(source_pt)
pc = o3.geometry.PointCloud()
pc.points = o3.utility.Vector3dVector(to_cpu(result))
pc.paint_uniform_color([0, 1, 0])
target.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([pc, target])
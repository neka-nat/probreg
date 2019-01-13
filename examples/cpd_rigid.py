import copy
import numpy as np
import open3d as o3
import transformations as trans
from probreg import cpd

source = o3.read_point_cloud("bunny.pcd")
source = o3.voxel_down_sample(source, voxel_size=0.01)
target = copy.deepcopy(source)
ans = trans.euler_matrix(*np.deg2rad([0.0, 0.0, 30.0]))
target.transform(ans)

params = cpd.registration_cpd(source, target)
print("result: ", params)
result = copy.deepcopy(source)
result.points = cpd.RigidCPD.transform(result.points, params)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])
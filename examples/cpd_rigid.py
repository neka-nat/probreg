import copy
import numpy as np
import open3d as o3
import transformations as trans
from probreg import cpd

source = o3.read_point_cloud("bunny.pcd")
source = o3.voxel_down_sample(source, voxel_size=0.005)
print(source)
target = copy.deepcopy(source)
tp = np.asarray(target.points)
target.points = o3.Vector3dVector(tp + 0.001 * np.random.randn(*tp.shape))
ans = trans.euler_matrix(*np.deg2rad([0.0, 0.0, 30.0]))
target.transform(ans)

params = cpd.registration_cpd(source, target)
rot = trans.identity_matrix()
rot[:3, :3] = params.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      params.scale, params.t)
result = copy.deepcopy(source)
result.points = cpd.RigidCPD.transform(result.points, params)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])
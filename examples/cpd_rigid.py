import copy
import numpy as np
import open3d as o3
import transformations as trans
from probreg import cpd

source = o3.read_point_cloud("bunny.pcd")
target = copy.deepcopy(source)
ans = trans.euler_matrix(*np.deg2rad([0.0, 0.0, 90.0]))
target.transform(ans)

res = cpd.registration_cpd(source, target, max_iteration=10)
print("result: ", res)
result = copy.deepcopy(source)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])
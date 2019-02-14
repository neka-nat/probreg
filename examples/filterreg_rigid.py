import copy
import numpy as np
import open3d as o3
import transformations as trans
from probreg import filterreg
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

tf_param, _, _ = filterreg.registration_filterreg(source, target, sigma2=0.01)
rot = trans.identity_matrix()
rot[:3, :3] = tf_param.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.scale, tf_param.t)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])
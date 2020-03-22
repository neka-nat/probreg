import numpy as np
import open3d as o3
import transformations as trans
from probreg import bcpd
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param, _, _ = bcpd.registration_bcpd(source, target,
                                        callbacks=cbs)
rot = trans.identity_matrix()
rot[:3, :3] = tf_param.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.scale, tf_param.t)
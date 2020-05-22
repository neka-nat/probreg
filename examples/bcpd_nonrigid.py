import numpy as np
import open3d as o3
import transformations as trans
from probreg import bcpd
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_nonrigid_3d('bunny-x.txt',
                                                             'bunny-y.txt', 0.1)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param = bcpd.registration_bcpd(source, target,
                                  callbacks=cbs)
rot = trans.identity_matrix()
rot[:3, :3] = tf_param.rigid_trans.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.rigid_trans.scale, tf_param.rigid_trans.t)
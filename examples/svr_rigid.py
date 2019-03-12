import numpy as np
import transformations as trans
from probreg import l2dist_regs
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd', n_random=0)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param = l2dist_regs.registration_svr(source, target, callbacks=cbs)
rot = trans.identity_matrix()
rot[:3, :3] = tf_param.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.scale, tf_param.t)
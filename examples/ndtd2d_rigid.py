import numpy as np
import transforms3d as t3d
from probreg import l2dist_regs
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd', n_random=0)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param = l2dist_regs.registration_ndtd2d(source, target, 0.05, callbacks=cbs)

print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
      tf_param.scale, tf_param.t)

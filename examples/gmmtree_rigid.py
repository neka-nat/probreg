import numpy as np
import transforms3d as t3d
from probreg import gmmtree
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd', n_random=0)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param, _ = gmmtree.registration_gmmtree(source, target, callbacks=cbs)

print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
      tf_param.scale, tf_param.t)

import numpy as np
import open3d as o3
import transforms3d as t3d
from probreg import cpd
from probreg import callbacks
import utils
import logging
log = logging.getLogger('probreg')
log.setLevel(logging.DEBUG)

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param, _, _ = cpd.registration_cpd(source, target,
                                      callbacks=cbs)

print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
      tf_param.scale, tf_param.t)

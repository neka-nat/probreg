import numpy as np
import transforms3d as t3d
from probreg import filterreg
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
objective_type = 'pt2pt'
tf_param, _, _ = filterreg.registration_filterreg(source, target,
                                                  objective_type=objective_type,
                                                  sigma2=None,
                                                  update_sigma2=True,
                                                  callbacks=cbs)

print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
      tf_param.scale, tf_param.t)

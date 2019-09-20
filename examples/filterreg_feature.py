import numpy as np
import transformations as trans
from probreg import filterreg
from probreg import features
from probreg import callbacks
import utils

source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd',
                                                          orientation=np.deg2rad([0.0, 0.0, 80.0]),
                                                          translation=np.array([0.5, 0.0, 0.0]),
                                                          n_random=0)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
objective_type = 'pt2pt'
tf_param, _, _ = filterreg.registration_filterreg(source, target,
                                                  objective_type=objective_type,
                                                  sigma2=1000, feature_fn=features.FPFH(),
                                                  callbacks=cbs)
rot = trans.identity_matrix()
rot[:3, :3] = tf_param.rot
print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
      tf_param.scale, tf_param.t)
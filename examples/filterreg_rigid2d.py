from probreg import filterreg
from probreg import callbacks
import matplotlib.pyplot as plt
import utils
import numpy as np

source, target = utils.prepare_source_and_target_nonrigid_2d('fish_source.txt',
                                                             'fish_target.txt')
cbs = [callbacks.Plot2DCallback(source, target)]
tf_param, _, _ = filterreg.registration_filterreg(source, target,
                                                  objective_type="pt2pt",
                                                  callbacks=cbs,
                                                  tf_init_params={"rot": np.identity(2), "t": np.zeros(2)})
plt.show()
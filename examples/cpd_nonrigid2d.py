from probreg import cpd
from probreg import callbacks
import matplotlib.pyplot as plt
import utils

source, target = utils.prepare_source_and_target_nonrigid_2d('fish_source.txt',
                                                             'fish_target.txt')
cbs = [callbacks.Plot2DCallback(source, target)]
tf_param, _, _ = cpd.registration_cpd(source, target, 'nonrigid',
                                      callbacks=cbs)
plt.show()
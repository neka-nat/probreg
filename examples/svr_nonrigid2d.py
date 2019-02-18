from probreg import l2dist_regs
from probreg import callbacks
import matplotlib.pyplot as plt
import utils

source, target = utils.prepare_source_and_target_nonrigid_2d('fish_source.txt',
                                                             'fish_target.txt')
cbs = [callbacks.Plot2DCallback(source, target)]
tf_param = l2dist_regs.registration_svr(source, target, 'nonrigid', callbacks=cbs)
plt.show()
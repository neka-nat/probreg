import numpy as np
from probreg import cpd
import matplotlib.pyplot as plt
import utils

source, target = utils.prepare_source_and_target_nonrigid_2d('fish_source.txt',
                                                             'fish_target.txt')
tf_param, _, _ = cpd.registration_cpd(source, target, 'nonrigid')
result = tf_param.transform(source)

plt.axis('equal')
plt.plot(source[:, 0], source[:, 1], 'ro', label='source')
plt.plot(target[:, 0], target[:, 1], 'g^', label='taget')
plt.plot(result[:, 0], result[:, 1], 'bo', label='result')
plt.legend()
plt.show()
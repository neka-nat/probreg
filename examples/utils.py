import copy
import numpy as np
import open3d as o3
import transformations as trans

def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       orientation=np.deg2rad([0.0, 0.0, 30.0])):
    source = o3.read_point_cloud(source_filename)
    source = o3.voxel_down_sample(source, voxel_size=0.005)
    print(source)
    target = copy.deepcopy(source)
    tp = np.asarray(target.points)
    target.points = o3.Vector3dVector(tp + noise_amp * np.random.randn(*tp.shape))
    ans = trans.euler_matrix(*orientation)
    target.transform(ans)
    return source, target

def prepare_source_and_target_nonrigid_2d(source_filename,
                                          target_filename):
    source = np.loadtxt(source_filename)
    target = np.loadtxt(target_filename)
    return source, target
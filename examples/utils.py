import copy
import numpy as np
import open3d as o3
import transformations as trans


def estimate_normals(pcd, params):
    if o3.__version__ >= '0.8.0.0':
        pcd.estimate_normals(search_param=params)
        pcd.orient_normals_to_align_with_direction()
    else:
        o3.estimate_normals(pcd, search_param=params)
        o3.orient_normals_to_align_with_direction(pcd)


def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       n_random=500,
                                       orientation=np.deg2rad([0.0, 0.0, 30.0]),
                                       translation=np.zeros(3),
                                       normals=False):
    source = o3.read_point_cloud(source_filename)
    source = o3.voxel_down_sample(source, voxel_size=0.005)
    print(source)
    target = copy.deepcopy(source)
    tp = np.asarray(target.points)
    np.random.shuffle(tp)
    rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))
    rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
    target.points = o3.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape), rands])
    ans = trans.euler_matrix(*orientation)
    ans[:3, 3] = translation
    target.transform(ans)
    if normals:
        estimate_normals(source, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        estimate_normals(target, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    return source, target

def prepare_source_and_target_nonrigid_2d(source_filename,
                                          target_filename):
    source = np.loadtxt(source_filename)
    target = np.loadtxt(target_filename)
    return source, target

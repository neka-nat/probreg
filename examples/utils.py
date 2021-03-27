import copy
import numpy as np
import open3d as o3
import transforms3d as t3d


def estimate_normals(pcd, params):
    pcd.estimate_normals(search_param=params)
    pcd.orient_normals_to_align_with_direction()


def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       n_random=500,
                                       orientation=np.deg2rad([0.0, 0.0, 30.0]),
                                       translation=np.zeros(3),
                                       voxel_size=0.005,
                                       normals=False):
    source = o3.io.read_point_cloud(source_filename)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    target = copy.deepcopy(source)
    tp = np.asarray(target.points)
    np.random.shuffle(tp)
    rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))
    rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
    target.points = o3.utility.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape), rands])
    ans = np.identity(4)
    ans[:3, :3] = t3d.euler.euler2mat(*orientation)
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


def prepare_source_and_target_nonrigid_3d(source_filename,
                                          target_filename,
                                          voxel_size=5.0):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(np.loadtxt(source_filename))
    target.points = o3.utility.Vector3dVector(np.loadtxt(target_filename))
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    print(target)
    return source, target
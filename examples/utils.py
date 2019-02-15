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


class Open3dVisualizerCallback(object):
    def __init__(self, source, target, save=False):
        self._vis = o3.Visualizer()
        self._vis.create_window()
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._source.paint_uniform_color([1, 0, 0])
        self._target.paint_uniform_color([0, 1, 0])
        self._result.paint_uniform_color([0, 0, 1])
        self._vis.add_geometry(self._source)
        self._vis.add_geometry(self._target)
        self._vis.add_geometry(self._result)
        self._cnt = 0

    def __del__(self):
        self._vis.destroy_window()

    def __call__(self, res):
        self._result.points = res.transformation.transform(self._source.points)
        self._vis.update_geometry()
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
        self._cnt += 1

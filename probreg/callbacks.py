import copy
import open3d as o3
import matplotlib.pyplot as plt


class Plot2DCallback(object):
    """Display the 2D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """
    def __init__(self, source, target, save=False,
                 keep_window=True):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._cnt = 0
        plt.axis('equal')
        plt.plot(self._source[:, 0], self._source[:, 1], 'ro', label='source')
        plt.plot(self._target[:, 0], self._target[:, 1], 'g^', label='taget')
        plt.plot(self._result[:, 0], self._result[:, 1], 'bo', label='result')
        plt.legend()
        plt.draw()

    def __call__(self, transformation):
        self._result = transformation.transform(self._source)
        plt.cla()
        plt.axis('equal')
        plt.plot(self._source[:, 0], self._source[:, 1], 'ro', label='source')
        plt.plot(self._target[:, 0], self._target[:, 1], 'g^', label='taget')
        plt.plot(self._result[:, 0], self._result[:, 1], 'bo', label='result')
        plt.legend()
        if self._save:
            plt.savefig('image_%04d.png' % self._cnt)
        plt.draw()
        plt.pause(0.001)
        self._cnt += 1


class Open3dVisualizerCallback(object):
    """Display the 3D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
        keep_window (bool, optional): If this flag is True,
            the drawing window blocks after registration is finished.
    """
    def __init__(self, source, target, save=False,
                 keep_window=True):
        self._vis = o3.Visualizer()
        self._vis.create_window()
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._keep_window = keep_window
        self._source.paint_uniform_color([1, 0, 0])
        self._target.paint_uniform_color([0, 1, 0])
        self._result.paint_uniform_color([0, 0, 1])
        self._vis.add_geometry(self._source)
        self._vis.add_geometry(self._target)
        self._vis.add_geometry(self._result)
        self._cnt = 0

    def __del__(self):
        if self._keep_window:
            self._vis.run()
        self._vis.destroy_window()

    def __call__(self, transformation):
        self._result.points = transformation.transform(self._source.points)
        self._vis.update_geometry()
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
        self._cnt += 1

import copy
import numpy as np
import open3d as o3
from probreg import filterreg
from probreg import callbacks
from probreg import transformation
from dq3d import dualquat, quat

n_points = 30
points = np.array([[i * 0.05, 0.0, 0.0] for i in range(n_points)])
tfs = [(quat(np.deg2rad(0.0), np.array([0.0, 0.0, 1.0])), np.array([0.0, 0.0, 0.0])),
       (quat(np.deg2rad(30.0), np.array([0.0, 0.0, 1.0])), np.array([0.0, 0.0, 0.3]))]
dqs = [dualquat(t[0], t[1]) for t in tfs]
ws = transformation.DeformableKinematicModel.SkinningWeight(n_points)
for i in range(n_points):
    ws['pair'][i][0] = 0
    ws['pair'][i][1] = 1
for i in range(n_points):
    ws['val'][i][0] = float(i) / n_points
    ws['val'][i][1] = 1.0 - float(i) / n_points
dtf = transformation.DeformableKinematicModel(dqs, ws)
tf_points = dtf.transform(points)

source = o3.geometry.PointCloud()
source.points = o3.utility.Vector3dVector(points)
target = o3.geometry.PointCloud()
target.points = o3.utility.Vector3dVector(tf_points)

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
reg = filterreg.DeformableKinematicFilterReg(cv(source), ws, 0.01)
reg.set_callbacks(cbs)
tf_param, _, _ = reg.registration(cv(target))

print(dqs)
print(tf_param.dualquats)
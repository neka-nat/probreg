import numpy as np
import open3d as o3

pcd = o3.read_point_cloud("bunny.pcd")
o3.draw_geometries([pcd])
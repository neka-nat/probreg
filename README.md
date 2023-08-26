# ![logo](https://raw.githubusercontent.com/neka-nat/probreg/master/images/logo.png)
[![Build status](https://github.com/neka-nat/probreg/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/neka-nat/probreg/actions/workflows/build-and-test.yaml/badge.svg)
[![PyPI version](https://badge.fury.io/py/probreg.svg)](https://badge.fury.io/py/probreg)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/probreg/badge/?version=latest)](https://probreg.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/probreg)](https://pepy.tech/project/probreg)

Probreg is a library that implements point cloud **reg**istration algorithms with **prob**ablistic model.

The point set registration algorithms using stochastic model are more robust than ICP(Iterative Closest Point).
This package implements several algorithms using stochastic models and provides a simple interface with [Open3D](http://www.open3d.org/).

## Core features

* Open3D interface
* Rigid and non-rigid transformation

## Algorithms

* Maximum likelihood when the target or source point cloud is observation data
    * [Coherent Point Drift (2010)](https://arxiv.org/pdf/0905.2635.pdf)
    * [Extended Coherent Point Drift (2016)](https://ieeexplore.ieee.org/abstract/document/7477719) (add correspondence priors to CPD)
    * [FilterReg (CVPR2019)](https://arxiv.org/pdf/1811.10136.pdf)
* Variational Bayesian inference
    * [Bayesian Coherent Point Drift (2020)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8985307)
* Distance minimization of two probabilistic distributions
    * [GMMReg (2011)](https://ieeexplore.ieee.org/document/5674050)
    * [Support Vector Registration (2015)](https://arxiv.org/pdf/1511.04240.pdf)
* Hierarchical Stocastic model
    * [GMMTree (ECCV2018)](https://arxiv.org/pdf/1807.02587.pdf)

### Transformations

| type | CPD | SVR, GMMReg | GMMTree | FilterReg | BCPD (experimental) |
|------|-----|-------------|---------|-----------|---------------------|
|Rigid | **Scale + 6D pose** | **6D pose** | **6D pose** | **6D pose** </br> (Point-to-point,</br> Point-to-plane,</br> FPFH-based)| - |
|NonRigid | **Affine**, **MCT** | **TPS** | - | **Deformable Kinematic** </br> (experimental) | **Combined model** </br> (Rigid + Scale + NonRigid-term)

### CUDA support
You need to install cupy.

```
pip install cupy
```

* [Rigid CPD](https://github.com/neka-nat/probreg/blob/master/examples/cpd_rigid_cuda.py)
* [Affine CPD](https://github.com/neka-nat/probreg/blob/master/examples/cpd_affine3d_cuda.py)

## Installation

You can install probreg using `pip`.

```
pip install probreg
```

Or install probreg from source.

```
git clone https://github.com/neka-nat/probreg.git --recursive
cd probreg
pip install -e .
```

## Getting Started

This is a sample code that reads a PCD file and calls CPD registration.
You can easily execute registrations from Open3D point cloud object and draw the results.

```py
import copy
import numpy as np
import open3d as o3
from probreg import cpd

# load source and target point cloud
source = o3.io.read_point_cloud('bunny.pcd')
source.remove_non_finite_points()
target = copy.deepcopy(source)
# transform target point cloud
th = np.deg2rad(30.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))
source = source.voxel_down_sample(voxel_size=0.005)
target = target.voxel_down_sample(voxel_size=0.005)

# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source, target)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])
```

## Resources

* [Documentation](https://probreg.readthedocs.io/en/latest/?badge=latest)

## Results

### Compare algorithms

| CPD | SVR | GMMTree | FilterReg |
|-----|-----|---------|-----------|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_rigid.gif" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/svr_rigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/gmmtree_rigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_rigid.gif" width="640"> |

### Noise test

| ICP(Open3D) | CPD | FilterReg |
|-------------|-----|-----------|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/icp_noise.gif" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_noise.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_noise.gif" width="640"> |

### Non rigid registration

| CPD | SVR | Filterreg | BCPD |
|-----|-----|-----------|------|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_nonrigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/svr_nonrigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_deformable.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/bcpd_nonrigid.gif" width="640"> |

### Feature based registration

| FPFH FilterReg |
|----------------|
|<img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_fpfh.gif" width="640"> |

### Time measurement

Execute an example script for measuring time.

```
OMP_NUM_THREADS=1 python time_measurement.py

# Results [s]
# ICP(Open3D):  0.02030642901081592
# CPD:  3.6435861150093842
# SVR:  0.5795929960149806
# GMMTree:  0.34479290700983256
# FilterReg:  0.039795294986106455
```

## Citing

```
@software{probreg,
    author = {{Kenta-Tanaka et al.}},
    title = {probreg},
    url = {https://probreg.readthedocs.io/en/latest/},
    version = {0.1.6},
    date = {2019-9-29},
}
```

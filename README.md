# ![logo](https://raw.githubusercontent.com/neka-nat/probreg/master/images/logo.png)
[![Build Status](https://travis-ci.org/neka-nat/probreg.svg?branch=master)](https://travis-ci.org/neka-nat/probreg)
[![Build status](https://ci.appveyor.com/api/projects/status/mdoohms52gnq6law?svg=true)](https://ci.appveyor.com/project/neka-nat/probreg)
[![PyPI version](https://badge.fury.io/py/probreg.svg)](https://badge.fury.io/py/probreg)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/probreg/badge/?version=latest)](https://probreg.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/probreg)](https://pepy.tech/project/probreg)

Probreg is a library that implements point cloud **reg**istration algorithms with **prob**ablistic model.

The point set registration algorithms using stochastic model are more robust than ICP(Iterative Closest Point).
This package implements several algorithms using stochastic models and provides a simple interface with [Open3D](http://www.open3d.org/).

## Core features

* Open3D interface
* Rigid and non-rigid transformation

## Algorithms

* Distance minimization of two point clouds
    * [Coherent Point Drift(2010)](https://arxiv.org/pdf/0905.2635.pdf)
    * [FilterReg(CVPR2019)](https://arxiv.org/pdf/1811.10136.pdf)
* Distance minimization of two probabilistic distributions
    * [GMMReg(2011)](https://ieeexplore.ieee.org/document/5674050)
    * [Support Vector Registration(2015)](https://arxiv.org/pdf/1511.04240.pdf)
* Hierarchical Stocastic model
    * [GMMTree(ECCV2018)](https://arxiv.org/pdf/1807.02587.pdf)

### Transformations

| type | CPD | SVR, GMMReg | GMMTree | FilterReg |
|------|-----|-------------|---------|-----------|
|Rigid | Scale + 6D pose | 6D pose | 6D pose | 6D pose|
|NonRigid | Affine, MCT | TPS | - | - |

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
source = o3.read_point_cloud('bunny.pcd')
target = copy.deepcopy(source)
# transform target point cloud
th = np.deg2rad(30.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))
# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source, target)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])
```

## Resources

* [Documentation](https://probreg.readthedocs.io/en/latest/?badge=latest)

## Results

### Compare algorithms

| CPD | SVR | GMMTree | FilterReg |
|-----|-----|---------|-----------|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_rigid.gif" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/svr_rigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/gmmtree_rigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_rigid.gif" width="640"> |

### Noise test

| ICP | CPD | FilterReg |
|-----|-----|-----------|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/icp_noise.gif" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_noise.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/filterreg_noise.gif" width="640"> |

### Non rigid registration

| CPD | SVR |
|-----|-----|
| <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/cpd_nonrigid.gif" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/probreg/master/images/svr_nonrigid.gif" width="640"> |

### Time measurement

Execute an example script for measuring time.

```
OMP_NUM_THREADS=1 python time_measurement.py

# Results [s]
# ICP:  0.01905073800298851
# CPD:  14.830138777004322
# SVR:  1.8208692720072577
# GMMTree:  0.48409615199489053
# FilterReg:  0.0498644180042902
```
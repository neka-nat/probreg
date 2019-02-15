# ![logo](images/logo.png)

Implementation of point cloud **reg**istration algorithms using **prob**ablistic model.

## Algorithms

* [Coherent Point Drift](https://arxiv.org/pdf/0905.2635.pdf)
* [GMMReg](https://ieeexplore.ieee.org/document/5674050)
* [Support Vector Registration](https://arxiv.org/pdf/1511.04240.pdf)
* [GMMTree](https://arxiv.org/pdf/1807.02587.pdf)
* [FilterReg](https://arxiv.org/pdf/1811.10136.pdf)

## Install

Install from source.

```
git clone https://github.com/neka-nat/probreg.git --recursive
cd probreg
pipenv install
pipenv shell
```

## Results

### Regid CPD

![rigid_cpd](images/cpd_rigid.gif)

### Non rigid CPD

![nonrigid_cpd](images/cpd_nonrigid.gif)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "math_utils.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_math, m) {
    Eigen::initParallel();

    m.def("squared_kernel", &squaredKernel);
    m.def("rbf_kernel", &rbfKernel);
    m.def("tps_kernel_2d", &tpsKernel2d);
    m.def("tps_kernel_3d", &tpsKernel3d);
    m.def("inverse_multiquadric_kernel", &inverseMultiQuadricKernel);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
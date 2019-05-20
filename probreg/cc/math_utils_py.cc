#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "math_utils.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_math, m) {
    Eigen::initParallel();

    m.def("squared_kernel", &squaredKernel);
    m.def("rbf_kernel", &rbfKernel);
    m.def("tps_kernel_2d", &tpsKernel2d);
    m.def("tps_kernel_3d", &tpsKernel3d);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
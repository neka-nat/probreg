#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "math_utils.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_math, m) {
    m.def("msn_all_combination", &meanSquareNormAllCombination);
    m.def("gaussian_kernel", &gaussianKernel);
    m.def("tps_kernel_2d", &tpsKernel2d);
    m.def("tps_kernel_3d", &tpsKernel3d);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
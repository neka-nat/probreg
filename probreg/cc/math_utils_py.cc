#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "math_utils.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_math, m) {
    m.def("mean_square_norm", &meanSquareNorm);
    m.def("gaussian_kernel", &gaussianKernel);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
#include "kabsch.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Geometry>

using namespace probreg;
namespace py = pybind11;

PYBIND11_MODULE(_kabsch, m) {
    m.def("kabsch", &computeKabsch);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
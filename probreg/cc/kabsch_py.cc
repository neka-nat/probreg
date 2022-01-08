#include "kabsch.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Geometry>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace probreg;
namespace py = pybind11;

PYBIND11_MODULE(_kabsch, m) {
    Eigen::initParallel();

    m.def("kabsch", &computeKabsch);
    m.def("kabsch2d", &computeKabsch2d);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
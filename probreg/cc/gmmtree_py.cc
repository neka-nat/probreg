#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gmmtree.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_gmmtree, m) {
    Eigen::initParallel();

    m.def("build_gmmtree", buildGmmTree);
    m.def("gmmtree_reg_estep", gmmTreeRegEstep);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
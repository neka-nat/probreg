#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gmmtree.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_gmmtree, m) {
    m.def("build_gmmtree", buildGmmTree);
    m.def("gmmtree_reg_estep", gmmTreeRegEstep);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
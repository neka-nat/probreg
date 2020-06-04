#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "ndt.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_ndt, m) {
    m.def("compute_ndt", &computeNdt);
    m.def("compute_objective_function", &computeObjectiveFunction);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
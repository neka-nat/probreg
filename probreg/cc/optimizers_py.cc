#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "optimizers.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_optimizers, m) {
    m.def("gauss_newton", &gaussNewton);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ifgt.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_ifgt, m) {
    py::class_<Ifgt>(m, "Ifgt")
        .def(py::init<Matrix, Float, Float>())
        .def("compute", &Ifgt::compute);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
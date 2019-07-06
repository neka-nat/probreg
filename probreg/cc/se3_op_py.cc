#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "se3_op.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_se3_op, m) {
    m.def("diff_from_twist", py::overload_cast<const Matrix3X&, const Vector&>(&diffFromTwist));
    m.def("diff_from_twist", py::overload_cast<const Matrix3X&>(&diffFromTwist));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
#include "point_to_plane.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace probreg;
namespace py = pybind11;

PYBIND11_MODULE(_pt2pl, m) {
    m.def("compute_twist_for_pt2pl", &computeTwistForPointToPlane);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
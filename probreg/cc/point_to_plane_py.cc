#include "point_to_plane.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace probreg;
namespace py = pybind11;

PYBIND11_MODULE(_pt2pl, m) {
    Eigen::initParallel();
    m.def("compute_twist_for_pt2pl", &computeTwistForPointToPlane);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
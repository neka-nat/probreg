#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "ifgt.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_ifgt, m) {
    Eigen::initParallel();

    py::class_<Ifgt>(m, "Ifgt").def(py::init<Matrix, Float, Float>()).def("compute", &Ifgt::compute);

    m.def("_kcenter_clustering", [](const Matrix& data, Integer num_clusters) {
        auto res = computeKCenterClustering(data, num_clusters, 1.0e-4);
        return res.cluster_index_;
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
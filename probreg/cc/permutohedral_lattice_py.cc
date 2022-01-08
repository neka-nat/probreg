#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "permutohedral.h"
#include "types.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_permutohedral_lattice, m) {
    py::class_<Permutohedral>(m, "Permutohedral")
        .def(py::init())
        .def("init", &Permutohedral::init)
        .def("get_lattice_size", &Permutohedral::getLatticeSize)
        .def("filter", [](const Permutohedral& ph, const probreg::Matrix& v, Integer start) {
            probreg::Matrix out = probreg::Matrix::Zero(v.rows(), v.cols());
            ph.compute(out, v, false, start);
            return out;
        });

    m.def("filter", [](const probreg::Matrix& p, const probreg::Matrix& v, bool with_blur) {
        assert(p.cols() == v.cols());
        probreg::Matrix out = probreg::Matrix::Zero(v.rows(), p.cols());
        Permutohedral ph;
        ph.init(p, with_blur);
        ph.compute(out, v);
        return out;
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
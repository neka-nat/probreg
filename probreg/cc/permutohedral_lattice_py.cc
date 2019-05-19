#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "permutohedral.h"
#include "types.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_permutohedral_lattice, m) {
    m.def("filter", [](const probreg::Matrix& p, const probreg::Matrix& v, bool with_blur) {
        assert(p.cols() == v.cols());
        probreg::Matrix out = probreg::Matrix::Zero(v.rows(), p.cols());
        Permutohedral ph;
        ph.init(p, with_blur);
        ph.compute(out, v);
        return out;
    });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
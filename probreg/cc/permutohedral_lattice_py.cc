#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "types.h"
#include "permutohedral.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_permutohedral_lattice, m) {
    m.def("filter",
          [](const Matrix& p, const Matrix& v) {
              assert(p.rows == v.rows());
              Matrix out = Matrix::Zero(p.rows(), v.cols());
              PermutohedralLattice::filter(p.data(), p.cols(),
                                           v.data(), v.cols(),
                                           p.rows(), out.data());
              return out;
          });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
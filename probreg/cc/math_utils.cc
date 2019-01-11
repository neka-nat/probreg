#include "math_utils.h"

using namespace probreg;

Float
probreg::meanSquareNorm(const Matrix& a, const Matrix& b) {
    Float ans = 0.0;
    for (Integer i = 0; i < b.rows(); ++i) ans += (a.rowwise() - b.row(i)).rowwise().squaredNorm().sum();
    return ans / (a.rows() * a.cols() * b.rows());
}
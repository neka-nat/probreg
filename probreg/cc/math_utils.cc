#include "math_utils.h"

using namespace probreg;

Float
probreg::meanSquareNormAllCombination(const Matrix& a, const Matrix& b) {
    Float ans = 0.0;
    for (Integer i = 0; i < b.rows(); ++i)
        ans += (a.rowwise() - b.row(i)).rowwise().squaredNorm().sum();
    return ans / (a.rows() * a.cols() * b.rows());
}

Matrix
probreg::gaussianKernel(const Matrix& x, Float beta) {
    Matrix g = Matrix::Zero(x.rows(), x.rows());
    for (Integer i = 0; i < x.rows(); ++i) {
        auto diff2 = (x(Eigen::seq(0, i), Eigen::all).rowwise() - x.row(i)).rowwise().squaredNorm();
        g(Eigen::seq(0, i), i) = (-diff2 / (2.0 * beta)).array().exp();
    }
    return g.selfadjointView<Eigen::Upper>();
}

Matrix
probreg::tpsKernel2d(const Matrix& x, const Matrix& y) {
    static const Float eps = 1.0e-9;
    Matrix k = Matrix::Zero(x.rows(), y.rows());
    for (Integer i = 0; i < y.rows(); ++i) {
        auto diff = (x.rowwise() - y.row(i)).rowwise().norm();
        k(Eigen::all, i) = (diff.array() > eps).select(diff.array().pow(2) * diff.array().log(), 0.0);
    }
    return k;
}

Matrix
probreg::tpsKernel3d(const Matrix& x, const Matrix& y) {
    Matrix k = Matrix::Zero(x.rows(), y.rows());
    for (Integer i = 0; i < y.rows(); ++i) {
        auto diff = (x.rowwise() - y.row(i)).rowwise().norm();
        k(Eigen::all, i) = -diff;
    }
    return k;
}
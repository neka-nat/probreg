#include "math_utils.h"

using namespace probreg;

Matrix
probreg::kernelBase(const Matrix& x, const Matrix& y, const func_type& fn) {
    Matrix k = Matrix::Zero(x.rows(), y.rows());
    for (Integer i = 0; i < y.rows(); ++i) {
        auto diff2 = (x.rowwise() - y.row(i)).rowwise().squaredNorm();
        k(Eigen::all, i) = fn(diff2);
    }
    return k;
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
probreg::squaredKernel(const Matrix& x, const Matrix& y) {
    return kernelBase(x, y);
}

Matrix
probreg::tpsKernel2d(const Matrix& x, const Matrix& y) {
    static const Float eps = 1.0e-9;
    return kernelBase(x, y,
      [] (const Vector& diff2) {return (diff2.array() > eps).select(diff2.array() * diff2.array().sqrt().log(), 0.0);});
}

Matrix
probreg::tpsKernel3d(const Matrix& x, const Matrix& y) {
   return kernelBase(x, y, [] (const Vector& diff2) {return -diff2.array().sqrt();});
}
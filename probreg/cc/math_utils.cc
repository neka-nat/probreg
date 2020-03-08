#include "math_utils.h"

using namespace probreg;

Matrix probreg::kernelBase(const Matrix& x, const Matrix& y, const func_type& fn) {
    Matrix k = Matrix::Zero(x.rows(), y.rows());
    #pragma omp parallel for
    for (Integer i = 0; i < y.rows(); ++i) {
        auto diff2 = (x.rowwise() - y.row(i)).rowwise().squaredNorm();
        k(Eigen::all, i) = fn(diff2);
    }
    return k;
}

Matrix probreg::squaredKernel(const Matrix& x, const Matrix& y) { return kernelBase(x, y); }

Matrix probreg::rbfKernel(const Matrix& x, const Matrix& y, Float beta) {
    return kernelBase(x, y, [&beta](const Vector& diff2) { return (-diff2 / (2.0 * beta)).array().exp(); });
}

Matrix probreg::tpsKernel2d(const Matrix& x, const Matrix& y) {
    static const Float eps = 1.0e-9;
    return kernelBase(x, y, [](const Vector& diff2) {
        return (diff2.array() > eps).select(diff2.array() * diff2.array().sqrt().log(), 0.0);
    });
}

Matrix probreg::tpsKernel3d(const Matrix& x, const Matrix& y) {
    return kernelBase(x, y, [](const Vector& diff2) { return -diff2.array().sqrt(); });
}

Matrix probreg::inverseMultiQuadricKernel(const Matrix& x, const Matrix& y, Float c) {
    return kernelBase(x, y, [&c](const Vector& diff2) { return 1.0 / (diff2.array() + c).sqrt(); });
}

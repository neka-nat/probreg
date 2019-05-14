#include "se3_op.h"

using namespace probreg;

Matrix18X probreg::diffFromTwist(const Matrix3X& points) {
    return diffFromTwist(points, Vector::Ones(points.cols()));
}

Matrix18X probreg::diffFromTwist(const Matrix3X& points, const Vector& weights) {
    assert(points.cols() == weights.size());
    Matrix18X ans = Matrix18X::Zero(18, points.cols());
    for (Integer i = 0; i < points.cols(); ++i) {
        const Vector& x = points.col(i);
        const Float x0 = x[0] * weights[i];
        const Float x1 = x[1] * weights[i];
        const Float x2 = x[2] * weights[i];
        const Float one = 1.0 * weights[i];
        ans(0 * 6 + 1, i) = x2;
        ans(0 * 6 + 2, i) = -x1;
        ans(0 * 6 + 3, i) = one;
        ans(1 * 6 + 0, i) = -x2;
        ans(1 * 6 + 2, i) = x0;
        ans(1 * 6 + 4, i) = one;
        ans(2 * 6 + 0, i) = x1;
        ans(2 * 6 + 1, i) = -x0;
        ans(2 * 6 + 5, i) = one;
    }
    return ans;
}

Matrix6 probreg::diffFromTwist2(const Matrix18X diff) {
    Matrix6 ans = Matrix6::Zero();
    for (Integer i = 0; i < diff.cols(); ++i) {
        const Matrix63 x = diff.col(i).reshaped(6, 3);
        ans(0, 0) += x(0, 1) * x(0, 1) + x(0, 2) * x(0, 2);
        ans(0, 1) += x(0, 2) * x(1, 2);
        ans(0, 2) += x(0, 1) * x(2, 1);
        ans(0, 4) += x(0, 1) * x(4, 1);
        ans(0, 5) += x(0, 2) * x(5, 2);
        ans(1, 1) += x(1, 0) * x(1, 0) + x(1, 2) * x(1, 2);
        ans(1, 2) += x(1, 0) * x(2, 0);
        ans(1, 3) += x(1, 0) * x(3, 0);
        ans(1, 5) += x(1, 2) * x(5, 2);
        ans(2, 2) += x(2, 0) * x(2, 0) + x(2, 1) * x(2, 1);
        ans(2, 3) += x(2, 0) * x(3, 0);
        ans(2, 4) += x(2, 1) * x(4, 1);
        ans(3, 3) += x(3, 0) * x(3, 0);
        ans(4, 4) += x(4, 1) * x(4, 1);
        ans(5, 5) += x(5, 2) * x(5, 2);
    }
    return ans.selfadjointView<Eigen::Upper>();
}
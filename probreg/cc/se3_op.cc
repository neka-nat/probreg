#include "se3_op.h"

using namespace probreg;

Matrix18X probreg::diffFromTwist(const Matrix3X& points) {
    return diffFromTwist(points, Vector3::Ones(points.cols()));
}

Matrix18X probreg::diffFromTwist(const Matrix3X& points, const Vector& weights){
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
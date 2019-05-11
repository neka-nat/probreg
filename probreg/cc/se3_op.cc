#include "se3_op.h"

using namespace probreg;

Matrix36Array probreg::diffFromTwist(const Matrix3X& points) {
    return diffFromTwist(points, Vector3::Ones(points.cols()));
}

Matrix36Array probreg::diffFromTwist(const Matrix3X& points, const Vector& weights){
    Matrix36Array ans(points.cols());
    for (Integer i = 0; i < points.cols(); ++i) {
        const Vector& x = points.col(i);
        ans[i].fill(0.0);
        Float x0 = x[0] * weights[i];
        Float x1 = x[1] * weights[i];
        Float x2 = x[2] * weights[i];
        Float one = 1.0 * weights[i];
        ans[i](0, 1) = x2;
        ans[i](0, 2) = -x1;
        ans[i](0, 3) = one;
        ans[i](1, 0) = -x2;
        ans[i](1, 2) = x0;
        ans[i](1, 4) = one;
        ans[i](2, 0) = x1;
        ans[i](2, 1) = -x0;
        ans[i](2, 5) = one;
    }
    return ans;
}
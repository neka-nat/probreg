#include "se3_op.h"

using namespace probreg;

Matrix36Array probreg::diffFromTwist(const Matrix3X& points){
    Matrix36Array ans(points.cols());
    for (Integer i = 0; i < points.cols(); ++i) {
        const Vector& x = points.col(i);
        ans[i].fill(0.0);
        ans[i](0, 1) = x[2];
        ans[i](0, 2) = -x[1];
        ans[i](0, 3) = 1.0;
        ans[i](1, 0) = -x[2];
        ans[i](1, 2) = x[0];
        ans[i](1, 4) = 1.0;
        ans[i](2, 0) = x[1];
        ans[i](2, 1) = -x[0];
        ans[i](2, 5) = 1.0;
    }
    return ans;
}
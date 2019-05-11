#ifndef __probreg_so3_op_h__
#define __probreg_so3_op_h__

#include "types.h"
#include <vector>

namespace probreg {

typedef std::vector<Matrix36> Matrix36Array;

Matrix36Array
diffFromTwist(const Matrix3X& points);

Matrix36Array
diffFromTwist(const Matrix3X& points, const Vector& weights);

}  // namespace probreg

#endif
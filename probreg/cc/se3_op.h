#ifndef __probreg_so3_op_h__
#define __probreg_so3_op_h__

#include "types.h"
#include <vector>

namespace probreg {

Matrix18X
diffFromTwist(const Matrix3X& points);

Matrix18X
diffFromTwist(const Matrix3X& points, const Vector& weights);

}  // namespace probreg

#endif
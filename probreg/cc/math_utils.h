#ifndef __probreg_math_utils_h__
#define __probreg_math_utils_h__

#include <functional>
#include "types.h"

namespace probreg {

typedef std::function<Vector(const Vector&)> func_type;

Matrix kernelBase(const Matrix& x, const Matrix& y, const func_type& fn = [](const Vector& diff2) {
    return diff2;
});

Matrix squaredKernel(const Matrix& x, const Matrix& y);

Matrix rbfKernel(const Matrix& x, const Matrix& y, Float beta);

Matrix tpsKernel2d(const Matrix& x, const Matrix& y);

Matrix tpsKernel3d(const Matrix& x, const Matrix& y);

Matrix inverseMultiQuadricKernel(const Matrix& x, const Matrix& y, Float c);

}  // namespace probreg

#endif
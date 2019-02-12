#ifndef __probreg_math_utils_h__
#define __probreg_math_utils_h__

#include "types.h"

namespace probreg
{

Float
meanSquareNormAllCombination(const Matrix& a, const Matrix& b);

Matrix
gaussianKernel(const Matrix& x, Float beta);

Matrix
tpsKernel2d(const Matrix& x, const Matrix& y);

Matrix
tpsKernel3d(const Matrix& x, const Matrix& y);

}

#endif
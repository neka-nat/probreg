#ifndef __probreg_math_utils_h__
#define __probreg_math_utils_h__

#include "types.h"

namespace probreg
{

Float
meanSquareNorm(const Matrix& a, const Matrix& b);

Matrix
gaussianKernel(const Matrix& a, Float beta);

}

#endif
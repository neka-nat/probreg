#ifndef __probreg_kabsch_h__
#define __probreg_kabsch_h__

#include "types.h"
#include <utility>

namespace probreg {

typedef std::pair<Matrix3, Vector3> KabschResult;

KabschResult computeKabsch(const Matrix3X& model,
                           const Matrix3X& target,
                           const Vector& weight);

}  // namespace probreg

#endif
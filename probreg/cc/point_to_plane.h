#ifndef __probreg_point_to_plane_h__
#define __probreg_point_to_plane_h__

#include "types.h"
#include <utility>

namespace probreg {

typedef std::pair<Vector6, Float> Pt2PlResult;

Pt2PlResult computeTwistForPointToPlane(const Matrix3X& model,
                                        const Matrix3X& target,
                                        const Matrix3X& target_normal,
                                        const Vector& weight);

}

#endif
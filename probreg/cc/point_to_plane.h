#ifndef __probreg_point_to_plane_h__
#define __probreg_point_to_plane_h__

#include "types.h"

namespace probreg {

Vector6 computeTwistForPointToPlane(const Matrix3X& model,
                                    const Matrix3X& target,
                                    const Matrix3X& target_normal,
                                    const Vector& weight);

}

#endif
#ifndef __probreg_point_to_plane_h__
#define __probreg_point_to_plane_h__

#include "types.h"
#include <utility>

namespace probreg {

typedef std::pair<Vector6, Float> Pt2PlResult;

Pt2PlResult computeTwistForPointToPlane(const MatrixX3& model,
                                        const MatrixX3& target,
                                        const MatrixX3& target_normal,
                                        const Vector& weight);

}

#endif
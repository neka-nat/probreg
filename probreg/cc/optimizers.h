#ifndef __probreg_optimizers_h__
#define __probreg_optimizers_h__

#include "types.h"
#include <functional>
#include <tuple>

namespace probreg {

typedef std::tuple<Matrix, Matrix, Vector> ResidualAmatBvec;
typedef std::pair<Vector, Matrix> GaussNewtonResult;
typedef std::function<ResidualAmatBvec(const Vector&)> gn_func_type;

GaussNewtonResult
gaussNewton(const Vector& x, const gn_func_type& fn, Float eps, Integer num_max_iteration = 50);

}  // namespace probreg

#endif
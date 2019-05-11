#include "optimizers.h"
#include <Eigen/Dense>
#include <iostream>

using namespace probreg;

GaussNewtonResult probreg::gaussNewton(const Vector& x, const gn_func_type& fn, Float eps, Integer num_max_iteration) {
    const Float eps2 = eps * eps;
    Vector xn = x;
    Matrix rx;
    for (Integer n = 0; n < num_max_iteration; ++n) {
        ResidualAmatBvec amat_bvec = fn(xn);
        const Vector dx = std::get<1>(amat_bvec).ldlt().solve(std::get<2>(amat_bvec));
        xn -= dx;
        rx = std::get<0>(amat_bvec);
        if (dx.squaredNorm() < eps2) {
            break;
        }
    }
    return std::make_pair(xn, rx);
}
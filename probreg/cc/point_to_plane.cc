#include "point_to_plane.h"
#include <Eigen/Dense>

using namespace probreg;

Vector6 probreg::computeTwistForPointToPlane(const Matrix3X& model,
                                             const Matrix3X& target,
                                             const Matrix3X& target_normal,
                                             const Vector& weight) {
    Matrix6 ata = Matrix6::Zero();
    Vector6 atb = Vector6::Zero();

    for (auto k = 0; k < model.cols(); ++k){
        const auto& vertex_k = model.col(k);
        const auto& target_k = target.col(k);
        const auto& normal_k = target_normal.col(k);
        const auto& weight_k = weight[k];
        const Float residual = normal_k.dot(vertex_k - target_k);
        const Vector6 jac = (Vector6() << vertex_k.cross(normal_k), normal_k).finished();
        for (Integer i = 0; i < 6; ++i) {
            for (Integer j = i; j < 6; ++j) {
                const Float jac_ij = weight_k * jac[i] * jac[j];
                ata(i, j) += jac_ij;
            }
        }
        for (Integer i = 0; i < 6; ++i) {
            const Float data = weight_k * (-residual * jac[i]);
            atb[i] += data;
        }
    }

    return ata.selfadjointView<Eigen::Upper>().ldlt().solve(atb);
}

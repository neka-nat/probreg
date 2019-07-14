#include "point_to_plane.h"
#include <Eigen/Dense>

using namespace probreg;

Pt2PlResult probreg::computeTwistForPointToPlane(const Matrix3X& model,
                                                 const Matrix3X& target,
                                                 const Matrix3X& target_normal,
                                                 const Vector& weight) {
    Matrix6 ata = Matrix6::Zero();
    Vector6 atb = Vector6::Zero();
    Float r_sum = 0.0;

    for (auto k = 0; k < model.cols(); ++k){
        const auto& vertex_k = model.col(k);
        const auto& target_k = target.col(k);
        const auto& normal_k = target_normal.col(k);
        const auto& weight_k = weight[k];
        const Float residual = normal_k.dot(target_k - vertex_k);
        const Vector6 jac = (Vector6() << vertex_k.cross(normal_k), normal_k).finished();
        ata += weight_k * jac * jac.transpose();
        atb += weight_k * residual * jac;
        r_sum += weight_k * weight_k * residual * residual;
    }
    return std::make_pair(ata.selfadjointView<Eigen::Upper>().ldlt().solve(atb), r_sum);
}

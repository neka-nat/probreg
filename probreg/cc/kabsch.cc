#include "kabsch.h"
#include <Eigen/Dense>

using namespace probreg;

KabschResult probreg::computeKabsch(const Matrix3X& model,
                                    const Matrix3X& target,
                                    const Vector& weight) {
    //Compute the center
    Vector3 model_center = Vector3::Zero();
    Vector3 target_center = Vector3::Zero();
    Float total_weight = 0.0f;
    for(auto i = 0; i < model.cols(); ++i) {
        const Float w_i = weight[i];
        total_weight += w_i;
        model_center += w_i * model.col(i);
        target_center += w_i * target.col(i);
    }
    Float divided_by = 1.0f / total_weight;
    model_center *= divided_by;
    target_center *= divided_by;

    //Centralize them
    //Compute the H matrix
    Float h_weight = 0.0f;
    Matrix3 hh = Matrix3::Zero();
    for(auto k = 0; k < model.cols(); ++k) {
        const auto& model_k = model.col(k);
        auto centralized_model_k = model_k - model_center;
        const auto& target_k = target.col(k);
        auto centralized_target_k = target_k - target_center;
        const Float this_weight = weight[k];
        h_weight += this_weight * this_weight;
        for(auto i = 0; i < 3; ++i) {
            for(auto j = 0; j < 3; ++j) {
                hh(i, j) += (this_weight * this_weight) * centralized_model_k[i] * centralized_target_k[j];
            }
        }
    }

    //Do svd
    hh /= h_weight;
    Eigen::JacobiSVD<Matrix3> svd(hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3 ss = Vector3::Ones(3);
    ss[2] = (svd.matrixU() * svd.matrixV()).determinant();
    Matrix3 r = svd.matrixV() * ss.asDiagonal() * svd.matrixU().transpose();

    //The translation
    Vector3 translation = target_center;
    translation -= r * model_center;

    return std::make_pair(r, translation);
}
#include "kabsch.h"
#include <Eigen/Dense>

using namespace probreg;

KabschResult probreg::computeKabsch(const MatrixX3& model,
                                    const MatrixX3& target,
                                    const Vector& weight) {
    //Compute the center
    Vector3 model_center = Vector3::Zero();
    Vector3 target_center = Vector3::Zero();
    Float total_weight = 0.0f;
    for(auto i = 0; i < model.rows(); ++i) {
        const Float w_i = weight[i];
        total_weight += w_i;
        model_center.noalias() += w_i * model.row(i);
        target_center.noalias() += w_i * target.row(i);
    }
    if (total_weight == 0) {
        return std::make_pair(Matrix3::Identity(), Vector3::Zero());
    }
    const Float divided_by = 1.0f / total_weight;
    model_center *= divided_by;
    target_center *= divided_by;

    //Centralize them
    //Compute the H matrix
    Float h_weight = 0.0f;
    Matrix3 hh = Matrix3::Zero();
    #pragma omp declare reduction(+ : Matrix3 : omp_out=omp_out+omp_in) initializer(omp_priv = omp_orig)
    #pragma omp parallel for reduction(+:h_weight) reduction(+:hh)
    for(auto k = 0; k < model.rows(); ++k) {
        const auto& model_k = model.row(k).transpose();
        auto centralized_model_k = model_k - model_center;
        const auto& target_k = target.row(k).transpose();
        auto centralized_target_k = target_k - target_center;
        const Float this_weight = weight[k];
        const Float w2 = this_weight * this_weight;
        const Matrix3 c2 = w2 * centralized_model_k * centralized_target_k.transpose();
        h_weight += w2;
        hh.noalias() += c2;
    }

    //Do svd
    hh /= h_weight;
    Eigen::JacobiSVD<Matrix3> svd(hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3 ss = Vector3::Ones(3);
    ss[2] = (svd.matrixU() * svd.matrixV()).determinant();
    const Matrix3 r = svd.matrixV() * ss.asDiagonal() * svd.matrixU().transpose();

    //The translation
    Vector3 translation = target_center;
    translation.noalias() -= r * model_center;

    return std::make_pair(r, translation);
}

KabschResult2d probreg::computeKabsch2d(const MatrixX2& model,
                                        const MatrixX2& target,
                                        const Vector& weight) {
    //Compute the center
    Vector2 model_center = Vector2::Zero();
    Vector2 target_center = Vector2::Zero();
    Float total_weight = 0.0f;
    for(auto i = 0; i < model.rows(); ++i) {
        const Float w_i = weight[i];
        total_weight += w_i;
        model_center.noalias() += w_i * model.row(i);
        target_center.noalias() += w_i * target.row(i);
    }
    if (total_weight == 0) {
        return std::make_pair(Matrix2::Identity(), Vector2::Zero());
    }
    const Float divided_by = 1.0f / total_weight;
    model_center *= divided_by;
    target_center *= divided_by;

    //Centralize them
    //Compute the H matrix
    Float h_weight = 0.0f;
    Matrix2 hh = Matrix2::Zero();
    #pragma omp declare reduction(+ : Matrix2 : omp_out=omp_out+omp_in) initializer(omp_priv = omp_orig)
    #pragma omp parallel for reduction(+:h_weight) reduction(+:hh)
    for(auto k = 0; k < model.rows(); ++k) {
        const auto& model_k = model.row(k).transpose();
        auto centralized_model_k = model_k - model_center;
        const auto& target_k = target.row(k).transpose();
        auto centralized_target_k = target_k - target_center;
        const Float this_weight = weight[k];
        const Float w2 = this_weight * this_weight;
        const Matrix2 c2 = w2 * centralized_model_k * centralized_target_k.transpose();
        h_weight += w2;
        hh.noalias() += c2;
    }

    //Do svd
    hh /= h_weight;
    Float angle = std::atan2((hh(0, 1) - hh(1, 0)), (hh(0, 0) + hh(1, 1)));
    Matrix2 r = Matrix2::Identity();
    r(0, 0) = r(1, 1) = std::cos(angle);
    r(0, 1) = -std::sin(angle);
    r(1, 0) = std::sin(angle);

    //The translation
    Vector2 translation = target_center;
    translation.noalias() -= r * model_center;

    return std::make_pair(r, translation);
}
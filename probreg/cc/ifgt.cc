#include <cmath>
#include <stdexcept>
#include "ifgt.h"

using namespace probreg;

namespace
{

Integer
nchoosek(Integer n, Integer k)
{
    if (k == 0) return 1;
    Integer n_k = n - k;
    if (k < n_k)
    {
        k = n_k;
        n_k = n - k;
    }

    Integer nchsk = 1; 
    for (Integer i = 1; i <= n_k; ++i)
    {
        nchsk *= (++k);
        nchsk /= i;
    }
    return nchsk;
}

Integer
chooseTruncationNumber(Integer num_dims, Float h, Float r, Float eps,
                       Float max_cluster_radius, Integer p_limit=300) {
    const Float h2 = h * h;
    const Float& rx = max_cluster_radius;
    const Float rx2 = rx * rx;
    Float error = std::numeric_limits<Float>::max();
    Float temp = 1.0;
    Integer p = 0;
    while ((error > eps) && (p <= p_limit)) {
        ++p;
        const Float b = std::min((rx + std::sqrt(rx2 + 2 * Float(p) * h2)) / 2.0, rx + r);
        const Float c = rx - b;
        temp *= 2 * rx * b / h2 / Float(p);
        error = temp * std::exp(-(c * c) / h2);
    }
    return p;
}

IfgtParameters
chooseIfgtParameters(Integer num_dims, Float h, Float eps,
                     Integer num_max_clusters, Integer p_limit=200)
{
    const Float r = std::min(std::sqrt(num_dims), h * std::sqrt(std::log(1.0 / eps)));
    Float complexity_min = std::numeric_limits<Float>::max();
    Integer num_clusters = 0;

    for (Integer i = 0; i < num_max_clusters; ++i) {
        const Float rx = std::pow(Float(i + 1), -1.0 / Float(num_dims));
        const Float n = std::min(Float(i + 1), std::pow(r / rx, (num_dims)));
        const Integer p = chooseTruncationNumber(num_dims, h, r, eps, rx, p_limit);
        Float complexity = i + 1 + std::log(Float(i + 1)) + (n + 1) * nchoosek(p - 1 + num_dims, num_dims);
        if (complexity < complexity_min) {
            complexity_min = complexity;
            num_clusters = i + 1;
        }
    }
    return {num_clusters, r};
}

Vector
computeMonomials(Integer num_dims, const Vector& d, Integer p, Integer p_max_total) {
    VectorXi heads = VectorXi::Zero(num_dims);
    Vector monomials = Vector::Ones(p_max_total);
    for (Integer k = 1, t = 1, tail = 1; k < p; ++k, tail = t) {
        for (Integer i = 0; i < num_dims; ++i) {
            Integer n = tail - heads[i];
            monomials(Eigen::seqN(t, n)) = d[i] * monomials(Eigen::seqN(heads[i], n));
            heads[i] = t;
            t += n;
        }
    }
    return monomials;
}

Vector
computeConstantSeries(Integer num_dims, Integer p, Integer p_max_total) {
    VectorXi heads = VectorXi::Zero(num_dims + 1);
    heads[num_dims] = std::numeric_limits<VectorXi::value_type>::max();
    VectorXi cinds = VectorXi::Zero(p_max_total);
    Vector monomials = Vector::Ones(p_max_total);

    for (Integer k = 1, t = 1, tail = 1; k < p; ++k, tail = t) {
        for (Integer i = 0; i < num_dims; ++i) {
            Integer n = tail - heads[i];
            auto rng = VectorXi::LinSpaced(n, heads[i], tail - 1);
            cinds(Eigen::seqN(t, n)).array() = (rng.array() < heads[i + 1]).select(cinds(Eigen::seqN(heads[i], n)).array() + 1, 1);
            monomials(Eigen::seqN(t, n)) = 2.0 * monomials(Eigen::seqN(heads[i], n));
            monomials(Eigen::seqN(t, n)).array() /= cinds(Eigen::seqN(t, n)).array().cast<Float>();
            heads[i] = t;
            t += n;
        }
    }
    return monomials;
}

}

Ifgt::Ifgt(const Matrix& source, Float h, Float eps) :source_(source), h_(h) {
    const Integer num_max_clusters = Integer(std::round(0.2 * 100 / h_));
    params_ = chooseIfgtParameters(source_.cols(), h_, eps, num_max_clusters);
    if (params_.num_clusters_ == 0) {
        throw std::runtime_error("Result of K center clustering is 0.");
    }
    cluster_ = computeKCenterClustering(source_, params_.num_clusters_, eps);
    const Float r = std::min(std::sqrt(source_.cols()), h_ * std::sqrt(std::log(1.0 / eps)));
    p_ = chooseTruncationNumber(source_.cols(), h_, r, eps, cluster_.max_cluster_radius_);
    p_max_total_ = nchoosek(p_ - 1 + source_.cols(), source_.cols());
    constant_series_ = computeConstantSeries(source_.cols(), p_, p_max_total_);
    ry2_ = (params_.cutoff_radius_ * Vector::Ones(params_.num_clusters_) + cluster_.cluster_radii_).array().pow(2).matrix();
}

Ifgt::~Ifgt() {}

Vector
Ifgt::compute(const Matrix& target, const Vector& weights) const {
    const Float h2 = h_ * h_;
    Matrix cmat = Matrix::Zero(params_.num_clusters_, p_max_total_);
    for (Integer i = 0; i < source_.rows(); ++i) {
        Vector dx = source_.row(i) - cluster_.cluster_centers_.row(cluster_.cluster_index_[i]);
        const Float distance = dx.array().pow(2).sum();
        dx /= h_;
        auto monomials = computeMonomials(source_.cols(), dx, p_, p_max_total_);
        Float f = weights[i] * std::exp(-distance / h2);
        cmat.row(cluster_.cluster_index_[i]) = f * monomials;
    }

    cmat.array().rowwise() *= constant_series_.transpose().array(); 
    Vector gvec = Vector::Zero(target.rows());
    for (Integer i = 0; i < source_.rows(); ++i) {
        for (Integer j = 0; j < params_.num_clusters_; ++j) {
            Vector dy = target.row(i) - cluster_.cluster_centers_.row(j);
            const Float distance = dy.array().pow(2).sum();
            if (distance > ry2_[j]) continue;
            dy /= h_;
            auto monomials = computeMonomials(source_.cols(), dy, p_, p_max_total_);
            const Float g = std::exp(-distance / h2);
            gvec[i] = cmat.row(j) * g * monomials;
        }
    }
    return gvec;
}
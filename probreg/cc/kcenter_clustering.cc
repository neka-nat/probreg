#include "kcenter_clustering.h"
#include <limits>

using namespace probreg;

ClusteringResult probreg::computeKCenterClustering(const Matrix& data,
                                                   Integer num_clusters,
                                                   Float eps,
                                                   Integer num_max_iteration) {
    auto idxs = (num_clusters * Vector::Random(num_clusters)).array().abs().cast<Integer>();
    Matrix cluster_centers = data(idxs, Eigen::all);
    Matrix temp_centers(num_clusters, data.cols());
    VectorXi counts(num_clusters);
    VectorXi labels(data.rows());
    Float p_err = 0.0;

    for (Integer n = 0; n < num_max_iteration; ++n) {
        counts.setZero();
        temp_centers.setZero();
        const Float err = updateClustering(data, cluster_centers, labels, counts, temp_centers);
        auto den = (counts.array() == 0).select(1, counts).replicate(1, data.cols()).array().cast<Float>();
        cluster_centers.noalias() = (temp_centers.array() / den).matrix();
        if (std::abs(err - p_err) < eps) break;
        p_err = err;
    }

    auto radii = calcRadii(data, cluster_centers, labels, num_clusters);
    return {radii.maxCoeff(), labels, cluster_centers, radii};
}

Float probreg::updateClustering(const Matrix& data,
                                const Matrix& cluster_centers,
                                VectorXi& labels,
                                VectorXi& counts,
                                Matrix& sum_members) {
    Float err = 0.0;
    #pragma omp parallel for
    for (Integer i = 0; i < data.rows(); ++i) {
        Float min_distance =
            (cluster_centers.rowwise() - data.row(i)).rowwise().squaredNorm().minCoeff(&labels[i]);
        #pragma omp critical
        {
            sum_members.row(labels[i]) += data.row(i);
            ++counts[labels[i]];
            err += min_distance;
        }
    }
    return err;
}

Vector probreg::calcRadii(const Matrix& data,
                          const Matrix& cluster_centers,
                          const VectorXi& labels,
                          Integer num_clusters) {
    const Vector distances = (data - cluster_centers(labels.array(), Eigen::all)).rowwise().norm();
    Vector radii = Vector::Zero(num_clusters);
    for (Integer i = 0; i < data.rows(); ++i) {
        radii[labels[i]] = std::max(radii[labels[i]], distances[i]);
    }
    return radii;
}
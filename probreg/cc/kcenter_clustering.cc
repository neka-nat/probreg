#include <limits>
#include "kcenter_clustering.h"

using namespace probreg;

ClusteringResult
probreg::computeKCenterClustering(const Matrix& data, Integer num_clusters,
                                  Float eps, Integer num_max_iteration)
{
    auto idxs = (num_clusters * Vector::Random(data.rows())).array().abs().cast<Integer>();
    Matrix cluster_centers = data(idxs, Eigen::all);
    auto num_cols = data.cols();
    auto num_rows = data.rows();
    Matrix temp_centers(num_clusters, num_cols);
    VectorXi counts(num_clusters);
    VectorXi labels(num_rows);
    Float err = 0.0;
    Float p_err = 0.0;

    for (Integer n = 0; n < num_max_iteration; ++n) {
        p_err = err;
        err = 0.0;
        counts.setZero();
        temp_centers.setZero();

        for (Integer i = 0; i < num_rows; ++i) {
            Float min_distance = (cluster_centers.rowwise() - data.row(i)).rowwise().squaredNorm().minCoeff(&labels[i]);
            temp_centers.row(labels[i]) += data.row(i);
            ++counts[labels[i]];
            err += min_distance;
        }
        cluster_centers = (temp_centers.array() / (counts.array() == 0).select(1, counts).array().cast<Float>()).matrix();
        if (std::abs(err - p_err) < eps) break;
    }

    Vector distances = (data - cluster_centers(labels.array(), Eigen::all)).rowwise().norm();
    Vector radii = Vector::Zero(num_clusters);
    for (Integer i = 0; i < num_rows; ++i) {
        radii[labels[i]] = std::max(radii[labels[i]], distances[i]);
    }

    return {radii.maxCoeff(), labels, cluster_centers, counts, radii};
}

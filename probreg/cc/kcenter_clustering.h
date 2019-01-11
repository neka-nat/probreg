#pragma once

#include "types.h"

namespace probreg
{

struct ClusteringResult
{
    Float max_cluster_radius_;
    VectorXi cluster_index_;
    Matrix cluster_centers_;
    Vector cluster_radii_;
};

ClusteringResult
computeKCenterClustering(const Matrix& data, Integer num_clusters,
                         Float eps, Integer num_max_iteration=100);

}
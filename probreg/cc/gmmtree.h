#ifndef __probreg_gmm_tree_h__
#define __probreg_gmm_tree_h__

#include <vector>
#include "types.h"

namespace probreg {

static const Integer N_NODE = 8;
typedef std::tuple<Float, Vector3, Matrix3> NodeParam;
typedef std::vector<NodeParam, Eigen::aligned_allocator<NodeParam> > NodeParamArray;

NodeParamArray buildGmmTree(const MatrixX3& points, Integer max_tree_level, Float lambda_s, Float lambda_d);

NodeParamArray gmmTreeEstep(const MatrixX3& points,
                            const NodeParamArray& nodes,
                            const VectorXi& parent_idx,
                            VectorXi& current_idx,
                            Integer max_tree_level);

void gmmTreeMstep(
    const NodeParamArray& params, Integer l, NodeParamArray& nodes, Integer n_points, Float lambda_d);

NodeParamArray gmmTreeRegEstep(const MatrixX3& points,
                               const NodeParamArray& nodes,
                               Integer max_tree_level,
                               Float lambda_c);

}  // namespace probreg

#endif
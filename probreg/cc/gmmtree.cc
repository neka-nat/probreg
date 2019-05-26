#define _USE_MATH_DEFINES
#include "gmmtree.h"
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace probreg;

namespace {
static const Float eps = 1.0e-15;

Float gaussianPdf(const Vector3& x, const Vector3& mu, const Matrix3& cov) {
    const Vector3 d = x - mu;
    const Float det = cov.determinant();
    if (det < eps) return 0.0;
    const Float c = 1.0 / (std::pow(det, 0.5) * std::pow(2.0 * M_PI, x.size() * 0.5));
    const Float ep = -0.5 * d.transpose() * cov.inverse() * d;
    return c * std::exp(ep);
}

Float logLikelihood(const NodeParamArray& nodes, const Matrix3X& points, Integer j0, Integer jn) {
    Float q = 0.0;
    for (Integer i = 0; i < points.cols(); ++i) {
        Float tmp = 0.0;
        for (Integer j = j0; j < jn; ++j) {
            if (std::get<0>(nodes[j]) < eps) continue;
            tmp += std::get<0>(nodes[j]) *
                   gaussianPdf(points.col(i), std::get<1>(nodes[j]), std::get<2>(nodes[j]));
        }
        q += std::log(std::max(tmp, eps));
    }
    return q;
}

Float complexity(const Matrix3& cov) {
    Eigen::SelfAdjointEigenSolver<Matrix3> es(cov);
    auto lmds = es.eigenvalues();
    std::sort(lmds.data(), lmds.data() + lmds.size(), std::greater<Float>());
    return lmds[2] / lmds.sum();
}

Integer child(Integer j) { return (j + 1) * N_NODE; }

Integer level(Integer l) { return N_NODE * (std::pow(N_NODE, l) - 1) / (N_NODE - 1); }

void accumulate(NodeParam& moments, Float gamma, const Vector& z) {
    if (gamma < eps) return;
    std::get<0>(moments) += gamma;
    std::get<1>(moments) += gamma * z;
    std::get<2>(moments) += gamma * z * z.transpose();
}

NodeParam mlEstimator(const NodeParam& moments, Integer n_points, Float lambda_d) {
    NodeParam node;
    std::get<0>(node) = std::get<0>(moments) / n_points;
    if (std::get<0>(moments) < lambda_d) {
        std::get<0>(node) = 0;
        std::get<1>(node).fill(0.0);
        std::get<2>(node) = Matrix3::Identity();
    } else {
        std::get<1>(node) = std::get<1>(moments) / std::get<0>(moments);
        std::get<2>(node) =
            std::get<2>(moments) / std::get<0>(moments) - std::get<1>(node) * std::get<1>(node).transpose();
    }
    return node;
}

}  // namespace

NodeParamArray probreg::buildGmmTree(const Matrix3X& points,
                                     Integer max_tree_level,
                                     Float lambda_s,
                                     Float lambda_d) {
    const Integer n_total = N_NODE * (1 - std::pow(N_NODE, max_tree_level)) / (1 - N_NODE);
    NodeParamArray nodes(n_total);
    auto idxs = (n_total * Vector::Random(n_total)).array().abs().cast<Integer>();
    Float sig2 = 0.0;
    for (Integer i = 0; i < points.cols(); ++i) {
        sig2 += (points.colwise() - points.col(i)).colwise().squaredNorm().sum();
    }
    sig2 /= points.cols() * points.cols() * points.rows() * N_NODE;
    for (Integer j = 0; j < n_total; ++j) {
        std::get<0>(nodes[j]) = 1.0 / N_NODE;
        std::get<1>(nodes[j]) = points.col(idxs[j]);
        std::get<2>(nodes[j]) = Matrix3::Identity() * sig2;
    }
    VectorXi parent_idx = -VectorXi::Ones(points.cols());
    VectorXi current_idx = VectorXi::Zero(points.cols());

    for (Integer l = 0; l < max_tree_level; ++l) {
        Float prev_q = 0.0;
        while (true) {
            const NodeParamArray params =
                gmmTreeEstep(points, nodes, parent_idx, current_idx, max_tree_level);
            gmmTreeMstep(params, l, nodes, points.cols(), lambda_d);
            const Float q = logLikelihood(nodes, points, level(l), level(l + 1));
            if (std::abs(q - prev_q) < lambda_s) {
                break;
            }
            prev_q = q;
        }
        parent_idx = current_idx;
    }
    return nodes;
}

NodeParamArray probreg::gmmTreeEstep(const Matrix3X& points,
                                     const NodeParamArray& nodes,
                                     const VectorXi& parent_idx,
                                     VectorXi& current_idx,
                                     Integer max_tree_level) {
    const Integer n_total = N_NODE * (1 - std::pow(N_NODE, max_tree_level)) / (1 - N_NODE);
    NodeParamArray moments(n_total);
    for (Integer j = 0; j < n_total; ++j) {
        std::get<0>(moments[j]) = 0.0;
        std::get<1>(moments[j]).fill(0.0);
        std::get<2>(moments[j]).fill(0.0);
    }

    for (Integer i = 0; i < points.cols(); ++i) {
        const Integer j0 = child(parent_idx[i]);
        Vector gamma = Vector::Zero(N_NODE);
        for (Integer j = j0; j < j0 + N_NODE; ++j) {
            gamma[j - j0] = std::get<0>(nodes[j]) *
                            gaussianPdf(points.col(i), std::get<1>(nodes[j]), std::get<2>(nodes[j]));
        }
        const Float den = gamma.sum();
        if (den > eps) {
            gamma /= den;
        }
        else {
            gamma.fill(0.0);
        }
        for (Integer j = j0; j < j0 + N_NODE; ++j) {
            accumulate(moments[j], gamma[j - j0], points.col(i));
        }
        Integer max_j;
        gamma.maxCoeff(&max_j);
        current_idx[i] = j0 + max_j;
    }
    return moments;
}

void probreg::gmmTreeMstep(
    const NodeParamArray& params, Integer l, NodeParamArray& nodes, Integer n_points, Float lambda_d) {
    const Integer lb = level(l);
    const Integer le = level(l + 1);

    for (Integer j = lb; j < le; ++j) {
        nodes[j] = mlEstimator(params[j], n_points, lambda_d);
    }
}

NodeParamArray probreg::gmmTreeRegEstep(const Matrix3X& points,
                                        const NodeParamArray& nodes,
                                        Integer max_tree_level,
                                        Float lambda_c) {
    const Integer n_total = N_NODE * (1 - std::pow(N_NODE, max_tree_level)) / (1 - N_NODE);
    NodeParamArray moments(n_total);
    for (Integer j = 0; j < n_total; ++j) {
        std::get<0>(moments[j]) = 0.0;
        std::get<1>(moments[j]).fill(0.0);
        std::get<2>(moments[j]).fill(0.0);
    }

    for (Integer i = 0; i < points.cols(); ++i) {
        Integer search_id = -1;
        Vector gamma = Vector::Zero(N_NODE);
        for (Integer l = 0; l < max_tree_level; ++l) {
            const Integer j0 = child(search_id);
            for (Integer j = j0; j < j0 + N_NODE; ++j) {
                gamma[j - j0] = std::get<0>(nodes[j]) *
                                gaussianPdf(points.col(i), std::get<1>(nodes[j]), std::get<2>(nodes[j]));
            }
            const Float den = gamma.sum();
            if (den > eps) {
                gamma /= den;
            } else {
                gamma.fill(0.0);
            }
            gamma.maxCoeff(&search_id);
            search_id += j0;
            if (complexity(std::get<2>(nodes[search_id])) <= lambda_c) break;
            accumulate(moments[search_id], gamma[search_id - j0], points.col(i));
        }
    }
    return moments;
}

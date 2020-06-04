#include "ndt.h"
#include <Eigen/Dense>
#include <iostream>

using namespace probreg;

namespace {

Matrix3 toRotation(const Vector6& p) {
    const Float sx = std::sin(p[3]);
    const Float cx = std::cos(p[3]);
    const Float sy = std::sin(p[4]);
    const Float cy = std::cos(p[4]);
    const Float sz = std::sin(p[5]);
    const Float cz = std::cos(p[5]);
    return (Matrix3() << cy * cz, -cy * sz, sy,
                         cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy,
                         sx * sz - cx * sy * cz, cx * sy * sz + sx * cz, cx * cy).finished();
}

std::tuple<Matrix3x6, Matrix18x6, Matrix3x18, Matrix18> computeDerivatives(const Vector3& x, const Matrix3& c1) {
    Matrix3x6 jest = Matrix3x6::Zero();
    Matrix18x6 hest = Matrix18x6::Zero();
    Matrix3x18 zest = Matrix3x18::Zero();
    Matrix18 zhest = Matrix18::Zero();

    jest(0, 4) = x[2];
    jest(0, 5) = -x[1];
    jest(1, 3) = -x[2];
    jest(1, 5) = x[0];
    jest(2, 3) = x[1];
    jest(2, 4) = -x[0];

    Matrix3 block;
    block << 0.0, -c1(0, 2), c1(0, 1),
             -c1(0, 2), -2 * c1(1, 2), -c1(2, 2) + c1(1, 1),
             c1(0, 1), -c1(2, 2) + c1(1, 1), 2 * c1(1, 2);
    zest.block<3, 3>(0, 9) = block;
    block << 2 * c1(0, 2), c1(1, 2), -c1(0, 0) + c1(2, 2),
             c1(1, 2), 0, -c1(0, 1),
             -c1(0, 0) + c1(2, 2), -c1(0, 1), -2 * c1(0, 2);
    zest.block<3, 3>(0, 12) = block;
    block << -2 * c1(0, 1), -c1(1, 1) + c1(0, 0), -c1(1, 2),
             -c1(1, 1) + c1(0, 0), 2 * c1(0, 1), c1(0, 2),
             -c1(1, 2), c1(0, 2), 0;
    zest.block<3, 3>(0, 15) = block;

    Vector3 a, b, c, d, e, f;
    a << 0, -x(1), -x(2);
    b << 0, x(0), 0;
    c << 0, 0, x(0);
    d << -x(0), 0, -x(2);
    e << 0, 0, x(1);
    f << -x(0), -x(1), 0;
    hest.block<3, 1>(9, 3) = a;
    hest.block<3, 1>(12, 3) = b;
    hest.block<3, 1>(15, 3) = c;
    hest.block<3, 1>(9, 4) = b;
    hest.block<3, 1>(12, 4) = d;
    hest.block<3, 1>(15, 4) = e;
    hest.block<3, 1>(9, 5) = c;
    hest.block<3, 1>(12, 5) = e;
    hest.block<3, 1>(15, 5) = f;
    block << 0, -c1(0, 1), -c1(0, 2),
             -c1(0, 1), 2 * c1(2, 2) - 2 * c1(1,1), -4 * c1(1, 2),
             -c1(0, 2), -4 * c1(1, 2), 2 * c1(1, 1) - 2 * c1(2, 2);
    zhest.block<3, 3>(9, 9) = block;
    block << 0, c1(0, 0) - c1(2, 2), c1(1, 2),
             c1(0, 0) - c1(2, 2), 2 * c1(0, 1), 2 * c1(0, 2),
             c1(1, 2), 2 * c1(0, 2), -2 * c1(0, 1);
    zhest.block<3, 3>(9, 12) = block;
    block << 0, c1(1, 2), c1(0, 0) - c1(1, 1),
             c1(1, 2), -2 * c1(0, 2), 2 * c1(0, 1),
             c1(0, 0) - c1(1, 1),  2 * c1(0, 1), 2 * c1(0, 2);
    zhest.block<3, 3>(9, 15) = block;
    block << 2 * c1(2, 2) - 2 * c1(0, 0), -c1(0, 1), -4 * c1(0, 2),
             -c1(0, 1), 0, -c1(1, 2),
             -4 * c1(0, 2), -c1(1, 2), 2 * c1(0, 0) - 2 * c1(2, 2);
    zhest.block<3, 3>(12, 12) = block;
    block << -2 * c1(1, 2), c1(0, 2), 2 * c1(0, 1),
            c1(0, 2), 0, c1(1, 1) - c1(0, 0),
            2 * c1(0, 1), c1(1, 1) - c1(0, 0), 2 * c1(1, 2);
    zhest.block<3, 3>(12, 15) = block;
    block << 2 * c1(1, 1) - 2 * c1(0, 0), -4 * c1(0, 1), -c1(0, 2),
            -4 * c1(0, 1), 2 * c1(0, 0) - 2 * c1(1, 1), -c1(1, 2),
            -c1(0, 2), -c1(1, 2), 0;
    zhest.block<3, 3>(15, 15)= block;
    zhest.block<3, 3>(12, 9) = zhest.block<3, 3>(9, 12);
    zhest.block<3, 3>(15, 9) = zhest.block<3, 3>(9, 15);
    zhest.block<3, 3>(15, 12)= zhest.block<3, 3>(12, 15);
    return std::make_tuple(jest, hest, zest, zhest);
}

void updateGradientHessian(Vector6& score_gradient,
                           Matrix6& hessian,
                           const Vector3& x,
                           const Matrix3& b,
                           Float likelihood,
                           const Matrix3x6& jest,
                           const Matrix18x6& hest,
                           const Matrix3x18& zest,
                           const Matrix18& zhest,
                           Float lfd2 = 0.05) {
    //vars for gradient
    Vector6 xtbj, xtbzbx, q;
    //vars for hessian
    Matrix6 xtbzbj, xtbh, xtbzbzbx, xtbzhbx;
    Vector3 tmp1, xtb;

    xtbj.setZero();
    xtbzbx.setZero();
    q.setZero();
    xtbzbj.setZero();
    xtbh.setZero();
    xtbzbzbx.setZero();
    xtbzhbx.setZero();
    tmp1.setZero();
    xtb.setZero();

    xtb = x.transpose() * b;
    xtbj = xtb.transpose() * jest;
    for (Integer i = 0; i < 6; i++) {
        tmp1 = xtb.transpose() * zest.block<3, 3>(0, 3 * i) * b;
        xtbzbx(i) = tmp1.dot(x);
        xtbzbj.col(i) = (tmp1.transpose() * jest).transpose();
        for(Integer j = 0; j < 6; j++) {
            xtbh(i, j) = xtb.transpose() * hest.block<3, 1>(3 * i, j);
            xtbzbzbx(i, j) = tmp1.transpose() * zest.block<3, 3>(0, 3 * j) * b * x;
            xtbzhbx(i, j) = xtb.transpose() * zhest.block<3, 3>(3 * i, 3 * j) * b * x;
        }
    }
    q = 2 * xtbj - xtbzbx;
    Float factor = -(lfd2 / 2) * likelihood;
    score_gradient += q * factor;

    hessian += factor*(2 * jest.transpose() * b * jest + 2 * xtbh - xtbzhbx  - 2 * xtbzbj.transpose()
                    -2 * xtbzbj + xtbzbzbx  + xtbzbzbx.transpose() - lfd2 * q * q.transpose() / 2);
}

}

NdtMap probreg::computeNdt(const MatrixX3& points, Float resolution) {
    NdtMap output;
    for (Integer i = 0; i < points.rows(); ++i) {
        const Vector3 pt = points.row(i);
        const Vector3i index = (pt / resolution).array().floor().cast<Integer>();
        VoxelIndex vidx = std::make_tuple(index[0], index[1], index[2]);
        auto itr = output.find(vidx);
        if (itr == output.end()) {
            output[vidx] = std::make_tuple(pt, pt * pt.transpose(), 1);
        } else {
            std::get<0>(itr->second) += pt;
            std::get<1>(itr->second) += pt * pt.transpose();
            std::get<2>(itr->second) += 1;
        }
    }
    NdtMap::iterator kv = output.begin();
    while (kv != output.end()) {
        const Integer n_pt = std::get<2>(kv->second);
        if (n_pt <= 3) {
            output.erase(kv++);
            continue;
        }
        std::get<0>(kv->second) /= n_pt;
        const Vector3 mean = std::get<0>(kv->second);
        std::get<1>(kv->second) /= n_pt;
        std::get<1>(kv->second) -= mean * mean.transpose();
        std::get<1>(kv->second) *= n_pt / (n_pt - 1);
        ++kv;
    }
    return output;
}

Objectives probreg::computeObjectiveFunction(const std::vector<Vector3>& mu1, const std::vector<Matrix3>& sigma1,
                                             const std::vector<Vector3>& mu2, const std::vector<Matrix3>& sigma2,
                                             const Vector6& p,
                                             Float lfd1, Float lfd2) {
    Integer n1 = mu1.size();
    Integer n2 = mu2.size();
    Float f = 0.0;
    const Vector3 t = p.head<3>();
    const Matrix3 rot = toRotation(p);
    Vector6 grad = Vector6::Zero();
    Matrix6 hesse = Matrix6::Zero();
    for (Integer i = 0; i < n1; ++i) {
        Matrix3x6 jest;
        Matrix18x6 hest;
        Matrix3x18 zest;
        Matrix18 zhest;
        std::tie(jest, hest, zest, zhest) = computeDerivatives(mu1[i], sigma1[i]);
        for (Integer j = 0; j < n2; ++j) {
            Vector3 transformed = mu1[i] - mu2[j];
            Matrix3 csum = sigma1[i] + sigma2[j];
            Matrix3 cinv;
            Float det;
            bool exists;
            csum.computeInverseAndDetWithCheck(cinv, det, exists, 1.0e-16);
            if (!exists) continue;
            Float l = transformed.dot(cinv * transformed);
            if (std::isnan(l) || std::isinf(l)) continue;
            Float sh = -lfd1 * std::exp(-lfd2 * l / 2.0);
            updateGradientHessian(grad, hesse, transformed, cinv, sh, jest, hest, zest, zhest, lfd2);
            f += sh;
        }
    }
    return Objectives(f, grad, hesse);
}

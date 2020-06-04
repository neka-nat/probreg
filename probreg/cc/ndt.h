#ifndef __probreg_ndt_h__
#define __probreg_ndt_h__

#include "types.h"
#include <vector>
#include <unordered_map>

namespace probreg {

typedef std::tuple<Integer, Integer, Integer> VoxelIndex;
typedef std::tuple<Vector3, Matrix3, Integer> NdtNode;
typedef std::tuple<Float, Vector6, Matrix6> Objectives;

struct hash {
    std::size_t operator() (const VoxelIndex& index) const {
        size_t seed = 0;
        auto elem1 = std::get<0>(index);
        seed ^= std::hash<Integer>()(elem1) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        auto elem2 = std::get<1>(index);
        seed ^= std::hash<Integer>()(elem2) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        auto elem3 = std::get<2>(index);
        seed ^= std::hash<Integer>()(elem3) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        return seed;
    }
};

typedef std::unordered_map<VoxelIndex, NdtNode, hash> NdtMap;

NdtMap computeNdt(const MatrixX3& points, Float resolution);

Objectives computeObjectiveFunction(const std::vector<Vector3>& mu1, const std::vector<Matrix3>& sigma1,
                                    const std::vector<Vector3>& mu2, const std::vector<Matrix3>& sigma2,
                                    const Vector6& p,
                                    Float lfd1 = 1.0, Float lfd2 = 0.05);

}

#endif
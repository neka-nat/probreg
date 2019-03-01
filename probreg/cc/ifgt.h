#ifndef __probreg_ifgt_h__
#define __probreg_ifgt_h__

#include "kcenter_clustering.h"

namespace probreg {

struct IfgtParameters {
    Integer num_clusters_;
    Float cutoff_radius_;
    Integer p_max_;
};

class Ifgt {
   public:
    Ifgt(const Matrix& source, Float h, Float eps);
    ~Ifgt();
    Vector compute(const Matrix& target, const Vector& weights) const;

   private:
    const Matrix source_;
    const Float h_;
    IfgtParameters params_;
    ClusteringResult cluster_;
    Integer p_;
    Integer p_max_total_;
    Vector constant_series_;
    Vector ry2_;
};

}  // namespace probreg

#endif
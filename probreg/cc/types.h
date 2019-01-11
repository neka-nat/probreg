#pragma once

#include <Eigen/Core>
//#define USE_DOUBLE

namespace probreg
{
#ifdef USE_DOUBLE
typedef double Float;
typedef Eigen::MatrixXd::Index Integer;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi VectorXi;
#else
typedef float Float;
typedef Eigen::MatrixXf::Index Integer;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::VectorXi VectorXi;
#endif
}
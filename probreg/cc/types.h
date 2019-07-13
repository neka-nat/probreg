#ifndef __probreg_types_h__
#define __probreg_types_h__

#include <Eigen/Core>
//#define USE_DOUBLE

namespace probreg {
#ifdef USE_DOUBLE
typedef double Float;
typedef int Integer;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Vector3d Vector3;
#else
typedef float Float;
typedef int Integer;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Vector3f Vector3;
#endif
typedef Eigen::Matrix<Float, 3, Eigen::Dynamic> Matrix3X;
typedef Eigen::Vector<Float, 6> Vector6;
typedef Eigen::Matrix<Float, 6, 6> Matrix6;
}  // namespace probreg

#endif
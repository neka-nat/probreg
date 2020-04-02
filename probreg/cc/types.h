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
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::Vector2d Vector2;
#else
typedef float Float;
typedef int Integer;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Vector3f Vector3;
typedef Eigen::Matrix2f Matrix2;
typedef Eigen::Vector2f Vector2;
#endif
typedef Eigen::Matrix<Float, Eigen::Dynamic, 3> MatrixX3;
typedef Eigen::Matrix<Float, Eigen::Dynamic, 2> MatrixX2;
typedef Eigen::Vector<Float, 6> Vector6;
typedef Eigen::Matrix<Float, 6, 6> Matrix6;
}  // namespace probreg

#endif
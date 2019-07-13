from __future__ import print_function
from __future__ import division
import numpy as np
import transformations as trans


def skew(x):
    """
    skew-symmetric matrix, that represent
    cross products as matrix multiplications.

    Args:
        x (numpy.ndarray): 3D vector.
    Returns:
        3x3 skew-symmetric matrix.
    """
    return np.array([[0.0, -x[2], x[1]],
                     [x[2], 0.0, -x[0]],
                     [-x[1], x[0], 0.0]])


def twist_trans(tw):
    """
    Linear approximation of transformation matrix
    using twist representation.

    Args:
        tw (numpy.ndarray): Twist vector.
    """
    return np.identity(3) + skew(tw[:3]), tw[3:]


def twist_mul(tw, rot, t, linear=False):
    if linear:
        tr, tt = twist_trans(tw)
        return np.dot(tr, rot), np.dot(t, tr.T) + tt
    else:
        twd = np.linalg.norm(tw[:3])
        if twd == 0.0:
            return rot, t + tw[3:]
        else:
            ntw = tw[:3] / twd
            c = np.cos(twd)
            s = np.sin(twd)
            tr = c * np.identity(3) + (1.0 - c) * np.outer(ntw, ntw) + s * skew(ntw)
            return np.dot(tr, rot), np.dot(t, tr.T) + tw[3:]


def diff_rot_from_quaternion(q):
    """Differencial rotation matrix from quaternion.

    dR(q)/dq = [dR(q)/dq0, dR(q)/dq1, dR(q)/dq2, dR(q)/dq3]

    Args:
        q (numpy.ndarray): Quaternion.
    """
    rot = trans.quaternion_matrix(q)[:3, :3]
    q2 = np.square(q)
    z = np.sum(q2)
    z2 = z * z
    d_rot = np.zeros((4, 3, 3))
    d_rot[0, 0, 0] = 4 * q[0] * (q2[2] + q2[3]) / z2
    d_rot[1, 0, 0] = 4 * q[1] * (q2[2] + q2[3]) / z2
    d_rot[2, 0, 0] = -4 * q[2] * (q2[1] + q2[0]) / z2
    d_rot[3, 0, 0] = -4 * q[3] * (q2[1] + q2[0]) / z2

    d_rot[0, 1, 1] = 4 * q[0] * (q2[1] + q2[3]) / z2
    d_rot[1, 1, 1] = -4 * q[1] * (q2[2] + q2[0]) / z2
    d_rot[2, 1, 1] = 4 * q[2] * (q2[1] + q2[3]) / z2
    d_rot[3, 1, 1] = -4 * q[3] * (q2[2] + q2[0]) / z2

    d_rot[0, 2, 2] = 4 * q[0] * (q2[1] + q2[2]) / z2
    d_rot[1, 2, 2] = -4 * q[1] * (q2[3] + q2[0]) / z2
    d_rot[2, 2, 2] = -4 * q[2] * (q2[1] + q2[2]) / z2
    d_rot[3, 2, 2] = 4 * q[3] * (q2[3] + q2[0]) / z2

    d_rot[0, 0, 1] = -2 * q[3] / z - 2 * q[0] * rot[0, 1] / z2
    d_rot[1, 0, 1] = 2 * q[2] / z - 2 * q[1] * rot[0, 1] / z2
    d_rot[2, 0, 1] = 2 * q[1] / z - 2 * q[2] * rot[0, 1] / z2
    d_rot[3, 0, 1] = -2 * q[0] / z - 2 * q[3] * rot[0, 1] / z2

    d_rot[0, 0, 2] = 2 * q[2] / z - 2 * q[0] * rot[0, 2] / z2
    d_rot[1, 0, 2] = 2 * q[3] / z - 2 * q[1] * rot[0, 2] / z2
    d_rot[2, 0, 2] = 2 * q[0] / z - 2 * q[2] * rot[0, 2] / z2
    d_rot[3, 0, 2] = 2 * q[1] / z - 2 * q[3] * rot[0, 2] / z2

    d_rot[0, 1, 0] = 2 * q[3] / z - 2 * q[0] * rot[1, 0] / z2
    d_rot[1, 1, 0] = 2 * q[2] / z - 2 * q[1] * rot[1, 0] / z2
    d_rot[2, 1, 0] = 2 * q[1] / z - 2 * q[2] * rot[1, 0] / z2
    d_rot[3, 1, 0] = 2 * q[0] / z - 2 * q[3] * rot[1, 0] / z2

    d_rot[0, 1, 2] = -2 * q[1] / z - 2 * q[0] * rot[1, 2] / z2
    d_rot[1, 1, 2] = -2 * q[0] / z - 2 * q[1] * rot[1, 2] / z2
    d_rot[2, 1, 2] = 2 * q[3] / z - 2 * q[2] * rot[1, 2] / z2
    d_rot[3, 1, 2] = 2 * q[2] / z - 2 * q[3] * rot[1, 2] / z2

    d_rot[0, 2, 0] = -2 * q[2] / z - 2 * q[0] * rot[2, 0] / z2
    d_rot[1, 2, 0] = 2 * q[3] / z - 2 * q[1] * rot[2, 0] / z2
    d_rot[2, 2, 0] = -2 * q[0] / z - 2 * q[2] * rot[2, 0] / z2
    d_rot[3, 2, 0] = 2 * q[1] / z - 2 * q[3] * rot[2, 0] / z2

    d_rot[0, 2, 1] = 2 * q[1] / z - 2 * q[0] * rot[2, 1] / z2
    d_rot[1, 2, 1] = 2 * q[0] / z - 2 * q[1] * rot[2, 1] / z2
    d_rot[2, 2, 1] = 2 * q[3] / z - 2 * q[2] * rot[2, 1] / z2
    d_rot[3, 2, 1] = 2 * q[2] / z - 2 * q[3] * rot[2, 1] / z2

    return d_rot

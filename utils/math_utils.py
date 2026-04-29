from scipy.spatial.transform import Rotation

import numpy as np
def matrix_to_pose(m):
    """
    Convert transformation matrix into pose (xyzrpy).

    """
    x = m[0][3] 
    y = m[1][3] 
    z = m[2][3] 
    rot_pose = np.eye(3)
    rot_pose = m[:3, :3]
    r = Rotation.from_matrix(rot_pose)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return x, y, z, roll, pitch, yaw

def pose_to_matrix(x,y,z,roll,pitch,yaw):
    """
    Convert pose (xyzrpy) into transformation matrix.

    """
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    matrix[:3, :3] = r.as_matrix()

    return matrix


def skew(vec):
    x, y, z = vec
    m = np.array([
        [0,-z,y],
        [z,0,-x],
        [-y,x,0]
    ])
    return m

def pose_to_jacobian(q, offset):
    """
    Convert pose (xyzrpy) into Jacobian matrix.

    """
    r = q.rotation.as_matrix()
    jacobian = np.hstack([
        np.eye(3),
        -r @ skew(offset)
    ])
    return jacobian
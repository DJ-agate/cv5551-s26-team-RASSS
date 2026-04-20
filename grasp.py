from scipy.spatial.transform import Rotation
import cv2, numpy, time, trimesh, math
from baseSDF import mesh

def se3t6dof(se3):
    """
    Convert transformation matrix into xyzrpy.

    """
    x = se3[0][3] * 1000
    y = se3[1][3] * 1000
    z = se3[2][3] * 1000

    rotation_matrix = se3[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return x,y,z,roll,pitch,yaw

def doftse3(x,y,z,roll,pitch,yaw):
    """
    Convert xyzrpy into transformation matrix.

    """
    matrix = numpy.eye(4)
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    
    R_x = numpy.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]
                    ])
                    
    R_y = numpy.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]
                    ])
                
    R_z = numpy.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                    ])
                    
    R = numpy.dot(R_z, numpy.dot( R_y, R_x ))

    matrix[:3, :3] = R

    return matrix

def query_SDF(q):
    """
    Query SDF value from mug STL file.

    parameters
    ----------
    q: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    """
    value = trimesh.proximity.signed_distance(q)
    return value

def grad_contact(q, d):
    """
    Calculate the distance cost between q and target grasp pose (UDP) 
    or how far is q from our ideal distance (SDF).

    parameters
    ----------
    q: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    d: float
        The ideal distance between gripper finger and mug rim/handle.
    """
    contact_cost = 2*math.abs(query_SDF(q) - d)
    return contact_cost

def grad_collision(q):
    """
    Calculate the collision cost of current gripper pose.

    parameters
    ----------
    q: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    """
    left_gripper = q.copy
    left_gripper[1][3] = left_gripper[1][3] + 0.001
    right_gripper = q.copy
    right_gripper[1][3] = right_gripper[1][3] + 0.001
    collision_cost = 2*max(0, query_SDF(left_gripper)) + 2*max(0, query_SDF(right_gripper))
    
    return collision_cost

def grad_smooth(q1, q0):
    """
    Calculate the smooth cost of current gripper pose comparing with last gripper pose.

    parameters
    ----------
    q1: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    q0: numpy.ndarray
        A 4x4 matrix representing the last gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    """
    smooth_cost = 0.5*numpy.dot((q1-q0).T,(q1-q0))
    return smooth_cost



def grasp_with_sdf(arm, gripper_pose, d=0.005):
    """
    Update grasp pose

    parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    gripper_pose: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    d: float
        The ideal distance between gripper finger and mug rim/handle
        d=0.005: Suppose the unit is meter and d is 5 mm
    """
    alpha = 0.001 
    cur_pose = arm.get_position(is_radian=True)
    cur_pose_m = se3t6dof(cur_pose)

    while cur_cost < prev_pose or not prev_pose: 
        if next_pose:
            cur_pose = next_pose
        conta_cost = grad_contact(cur_pose_m, d)  
        colli_cost = grad_collision(cur_pose_m)
        if prev_pose:
            smo_cost = grad_smooth(cur_cost, prev_pose)
            cur_cost = conta_cost + colli_cost + smo_cost
        else:
            cur_cost = conta_cost + colli_cost + smo_cost
        next_pose = cur_pose + alpha*cur_cost
        prev_pose = cur_pose
        arm.set_position(next_pose)
 
    return cur_cost







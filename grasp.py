from scipy.spatial.transform import Rotation
import cv2, numpy, time, trimesh, math
from baseSDF import mesh

# TODO
# Transform matrix from robot frame to model frame
T_ROBOT_MODEL = numpy.eye(4)

def matrix_to_pose(m):
    """
    Convert transformation matrix into pose (xyzrpy).

    """
    x = m[0][3] 
    y = m[1][3] 
    z = m[2][3] 

    rotation_matrix = m[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return x, y, z, roll, pitch, yaw

def pose_to_matrix(x,y,z,roll,pitch,yaw):
    """
    Convert pose (xyzrpy) into transformation matrix.

    """
    matrix = numpy.eye(4)
    matrix[:3, 3] = [x, y, z]
    
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    matrix[:3, :3] = r.as_matrix()

    return matrix

def query_SDF_point(p):
    """
    Query SDF value from mug STL file using a point.

    parameters
    ----------
    p: numpy.ndarray
        A (n,3) array representing a point in the model frame.
        All translational units in this matrix are in meters ???
    """
    p = numpy.asarray(p, dtype=float).reshape(1, 3)
    return trimesh.proximity.signed_distance(mesh, p)[0]

def query_SDF(q):
    """
    Query SDF value from mug STL file using a pose.

    parameters
    ----------
    q: numpy.ndarray
        A 4x4 matrix representing a pose in the model frame.
        All translational units in this matrix are in meters ???
    """
    
    # x = q[0][3]
    # y = q[1][3]
    # z = q[2][3]
    # point = numpy.array([[x,y,z]])
    # value = trimesh.proximity.signed_distance(mesh,point)[0]
    # return value
    return query_SDF_point(q[:3, 3])

def get_finger_points(q, offset=0.013):
    left_gripper  = q[:3, :3] @ numpy.array([0,  offset, 0]) + q[:3, 3]
    right_gripper = q[:3, :3] @ numpy.array([0, -offset, 0]) + q[:3, 3]
    return left_gripper, right_gripper

def get_contact_cost(q, d):
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
    # contact_cost = (query_SDF(q) - d)**2
    
    left_gripper, right_gripper = get_finger_points(q)
    contact_cost = (query_SDF_point(left_gripper) - d)**2 + (query_SDF_point(right_gripper) - d)**2
    return contact_cost

def get_collision_cost(q):
    """
    Calculate the collision cost of current gripper pose.

    parameters
    ----------
    q: numpy.ndarray
        A 4x4 matrix representing the current gripper pose in the model frame.
        All translational units in this matrix are in meters ???
    """
    left_gripper, right_gripper = get_finger_points(q)
    collision_cost = max(0, query_SDF_point(left_gripper))**2 + max(0, query_SDF_point(right_gripper))**2
    
    return collision_cost



def get_smooth_cost(cur_vec, prev_vec):
    """
    Calculate the smooth cost of current gripper pose comparing with last gripper pose.

    parameters
    ----------
    cur_vec: numpy.ndarray
        current xyzrpy
    prev_vec: numpy.ndarray
        privious xyzrpy
    """
    dp = cur_vec[:3] - prev_vec[:3]
    dr = cur_vec[3:] - prev_vec[3:]
    # Translation and rotation have different units (trans: meter, rot: rad)
    # Thus using different weights
    w1 = 1.0
    w2 = 0.05
    smooth_cost = 0.5*(w1*dp @ dp + w2*dr @ dr)
    # smooth_cost = 0.5*numpy.dot((cur_vec-prev_vec).T,(cur_vec-prev_vec))
    return smooth_cost

def total_cost(cur_vec, prev_vec, d, t_robot_model=T_ROBOT_MODEL):
    """
    Calculate total cost, including contact_cost, collision_cost and smooth_cost.

    parameters
    ----------
    cur_vec: numpy.ndarray
        current xyzrpy
    prev_vec: numpy.ndarray
        privious xyzrpy
    d: float
        The ideal distance between gripper finger and mug rim/handle
        d=0.005: Suppose the unit is meter and d is 5 mm
    t_robot_model: numpy.narray
        A 4x4 transformation matrix representing the transformation 
        from robot frame to model frame.

    """
    q_robot = pose_to_matrix(*cur_vec)
    q_model = t_robot_model @ q_robot # TODO: is it right?

    conta_cost = get_contact_cost(q_model, d)  
    colli_cost = get_collision_cost(q_model)
    smo_cost = get_smooth_cost(cur_vec, prev_vec)
    
    # TODO: may modify weights
    return conta_cost + colli_cost + 0.1*smo_cost


def get_grad(cur_vec, prev_vec, d, t_robot_model=T_ROBOT_MODEL, eps=1e-4):
    """
    Get gradient descent (forward difference)
    """
    grad = numpy.zeros(6)
    init_cost = total_cost(cur_vec, prev_vec, d, t_robot_model)
    
    eps_t = 1e-4 
    eps_r = 1e-3
    for i in range(6):
        perturbed_vec = cur_vec.copy()
        if i<3:
            eps = eps_t
            perturbed_vec[i] += eps
            
        else:
            eps = eps_r
            perturbed_vec[i] += eps
        
        cur_cost = total_cost(perturbed_vec, prev_vec, d, t_robot_model)
        grad[i] = (cur_cost - init_cost)/eps

    return grad

# def get_grad(cur_vec, prev_vec, d, t_robot_model=T_ROBOT_MODEL):
#     """
#     Get gradient descent (central difference, more precise than forward difference)
#     """
#     grad = numpy.zeros(6)
#     eps_t = 1e-4
#     eps_r = 1e-3

#     for i in range(6):
#         eps = eps_t if i < 3 else eps_r

#         vec_p = cur_vec.copy()
#         vec_m = cur_vec.copy()
#         vec_p[i] += eps
#         vec_m[i] -= eps

#         cost_p = total_cost(vec_p, prev_vec, d, t_robot_model)
#         cost_m = total_cost(vec_m, prev_vec, d, t_robot_model)

#         grad[i] = (cost_p - cost_m) / (2 * eps)

#     return grad

def grasp_with_sdf(arm, gripper_pose, d=0.005, t_robot_model=T_ROBOT_MODEL):
    """
    Update grasp pose using gradient descent 

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
    init_vec = numpy.array(matrix_to_pose(gripper_pose))
    cur_vec = init_vec.copy()
    prev_vec = init_vec.copy()

    # May add Adam optimizer later
    # optimizer = AdamOptimizer(alpha=0.01)

    # alpha = 0.001 
    alpha_t = 0.001
    alpha_r = 0.01
    max_iter = 200
    prev_cost = total_cost(cur_vec, prev_vec, d, t_robot_model)
    for i in range(max_iter):
        grad = get_grad(cur_vec, prev_vec, d, t_robot_model)

        # Normalization
        # grad_norm = numpy.linalg.norm(grad)
        # if grad_norm > 1e-8:
        #     grad = grad / grad_norm
        
        # next_vec = optimizer.step(cur_vec, grad)
        
        next_vec = cur_vec.copy()
        next_vec[:3] = cur_vec[:3] - alpha_t * grad[:3]
        next_vec[3:] = cur_vec[3:] - alpha_r * grad[3:]
        # next_vec = cur_vec - alpha * grad
        cur_cost = total_cost(next_vec, cur_vec, d, t_robot_model)
        
        if abs(prev_cost - cur_cost) < 1e-4:
            print(f"Converged at iter {i}")
            cur_vec = next_vec
            prev_cost = cur_cost
            break

        # prev_vec = cur_vec
        # prev_cost = cur_cost
        # cur_vec = next_vec
        if cur_cost < prev_cost:
            prev_vec = cur_vec
            cur_vec = next_vec
            prev_cost = cur_cost
        else:
            print(f"iter {i}: cost increased, stop or reduce step")
            break

        q_robot = pose_to_matrix(*cur_vec)
        q_model = t_robot_model @ q_robot
        left, right = get_finger_points(q_model)

        left_sdf = query_SDF_point(left)
        right_sdf = query_SDF_point(right)

        print(f"iter {i}: left={left_sdf:.6f}, right={right_sdf:.6f}, total={prev_cost:.6f}")

    print("Final pose:", cur_vec)
    print("Final cost:", prev_cost)
    tx, ty, tz, tr, tp, tyaw = cur_vec
    # Use mm as unit of robot motion
    arm.set_position(tx*1000, ty*1000, tz*1000, tr, tp, tyaw, is_radian=True)

        
    return cur_vec, prev_cost

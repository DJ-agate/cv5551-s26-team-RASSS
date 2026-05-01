from scipy.spatial.transform import Rotation
import cv2, time, trimesh, math
import numpy as np

mesh = trimesh.load_mesh("Mug_w_tags.stl")
print(mesh.extents)
mesh.apply_scale(0.02)
print(mesh.extents)
# TODO
T_ROBOT_MODEL = np.eye(4)


class Cost():
    def __init__(self, env):
        self.env = env #this is where the voxel sdf is calculated


    

    def query_SDF(self, q):
        """
        Query SDF values

        Needs to be called from obstalce cost

        parameters
        """
        # likely need to grab array of distances and gradients for a trajectory
        x = q[0][3]
        y = q[1][3]
        z = q[2][3]
        point = np.array([[x,y,z]])

        #Pseudo code to be verified
        # need to grab values from the computed field,
        value = self.env.sdf_voxel[x,y,z]
        gradient = self.env.sdf_gradient[x,y,z]
        return value, gradient


    def get_collision_loss(self, ei ,start, end):
        #ei = trajectory
        """
        Calculate the collision loss of current trajectory.

        parameters
        ----------
        q: numpy.ndarray
            A 4x4 matrix representing the current gripper pose in the model frame.
            All translational units in this matrix are in meters ???
        """
        left_gripper = q.copy()
        left_gripper[1][3] = left_gripper[1][3] + 0.013
        right_gripper = q.copy()
        right_gripper[1][3] = right_gripper[1][3] - 0.013
        collision_cost = 2*max(0, query_SDF(left_gripper)) + 2*max(0, query_SDF(right_gripper))
        
        return collision_cost

    def get_smooth_loss(self, ei, start, end):
        """
        Calculate the smooth cost of current gripper pose comparing with last gripper pose.

        parameters
        ----------
        cur_vec: numpy.ndarray
            current xyzrpy
        prev_vec: numpy.ndarray
            previous xyzrpy
        """
        SMOOTH_WEIGHT = 0.1

        smoothness_grad *= SMOOTH_WEIGHT
        return smoothness_loss, smoothness_grad

    def get_obstacle_cost():
        pass

    def total_loss(self, trajectory):


        q_robot = pose_to_matrix(*cur_vec)
        q_model = np.dot(t_robot_model, q_robot)


        smooth_loss, smooth_grad = self.get_smooth_cost(trajectory.data, trajectory.start, trajectory.end)

        obs_loss, obs_obs_grad, coll_pts, coll = self.get_obstacle_cost(trajectory)


        return
    def total_cost(self, cur_vec, prev_vec, d, t_robot_model=T_ROBOT_MODEL):
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
        q_model = numpy.dot(t_robot_model, q_robot)

        conta_cost = get_contact_cost(q_model, d)
        colli_cost = get_collision_cost(q_model)
        smo_cost = get_smooth_cost(cur_vec, prev_vec)

        return conta_cost + colli_cost + 0.1*smo_cost

    # def get_grad(self, cur_vec, prev_vec, d, eps=1e-6, t_robot_model=T_ROBOT_MODEL):
    #     """
    #     Get gradient descent 
    #     """
    #     grad = numpy.zeros(6)
    #     init_cost = total_cost(cur_vec, prev_vec, d, t_robot_model)
    #     #print("eps: ", eps)
    #     for i in range(6):
    #         perturbed_vec = cur_vec.copy()
    #         perturbed_vec[i] += eps
            
    #         cur_cost = total_cost(perturbed_vec, prev_vec, d, t_robot_model)
    #         grad[i] = (cur_cost - init_cost)/eps

    #     return grad


    # def grasp_with_sdf(self, gripper_pose, d=0.005, t_robot_model=T_ROBOT_MODEL):
    #     """
    #     Update grasp pose using gradient descent 

    #     parameters
    #     ----------
    #     arm : xarm.wrapper.XArmAPI
    #         The initialized XArm API object controlling the Lite6 robot.
    #     gripper_pose: numpy.ndarray
    #         A 4x4 matrix representing the current gripper pose in the model frame.
    #         All translational units in this matrix are in meters ???
    #     d: float
    #         The ideal distance between gripper finger and mug rim/handle
    #         d=0.005: Suppose the unit is meter and d is 5 mm
    #     """
    #     init_vec = numpy.array(matrix_to_pose(gripper_pose))
    #     cur_vec = init_vec.copy()
    #     prev_vec = init_vec.copy()

    #     alpha = 0.005
    #     # May add Adam optimizer later
    #     # optimizer = AdamOptimizer(alpha=0.01)

    #     max_iter = 200
    #     prev_cost = total_cost(cur_vec, prev_vec, d, t_robot_model=t_robot_model)
    #     for i in range(max_iter):
    #         grad = get_grad(cur_vec, init_vec, d, eps=1e-6, t_robot_model=t_robot_model)
    #         # next_vec = optimizer.step(cur_vec, grad)
    #         next_vec = cur_vec - alpha * grad
    #         cur_cost = total_cost(next_vec, init_vec, d, t_robot_model=t_robot_model)
    #         if abs(prev_cost - cur_cost) < 1e-4:
    #             print(f"Converged at iter {i}")
    #             break
    #         prev_cost = cur_cost
    #         cur_vec = next_vec

    #         tx, ty, tz, tr, tp, tyaw = cur_vec
    #         print(cur_vec)
    #         # Use mm as unit of robot motion
    #         #arm.set_position(tx*1000, ty*1000, tz*1000, tr, tp, tyaw, is_radian=True)

            
    #     return prev_cost
    # def matrix_to_pose(self, m):
    #     """
    #     Convert transformation matrix into pose (xyzrpy).

    #     """
    #     x = m[0][3] 
    #     y = m[1][3] 
    #     z = m[2][3]
    #     rotation_matrix = m[:3, :3]
    #     r = Rotation.from_matrix(rotation_matrix)
    #     roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    #     return x, y, z, roll, pitch, yaw

    # def pose_to_matrix(self, x,y,z,roll,pitch,yaw):
    #     """
    #     Convert pose (xyzrpy) into transformation matrix.

    #     """
    #     matrix = numpy.eye(4)
    #     matrix[:3, 3] = [x, y, z]
        
    #     r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        
    #     matrix[:3, :3] = r.as_matrix()

    #     return matrix






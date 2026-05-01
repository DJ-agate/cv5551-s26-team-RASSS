#based on CHOMP and NGDF

import numpy as np
import trimesh
import open3d as o3d
import scipy
# import quaternion as quat

from scipy.spatial.transform import RigidTransform, Rotation, Slerp
from copy import copy
from utils.math_utils import matrix_to_pose

from open3d.t.geometry import RayCastingScene

'''
Description: Optimizes a trajectory

'''
class objective_optimizer:
    '''
    inputs:
    q_endeffector: initial position of endeffector, quaternion
    q_grasps: ndarray of candidate grasps
    obj_meshes: list of object meshes
    '''    
    def __init__(self, q_endeffector, q_grasps, obj_meshes, obj_poses):
        self.q_start = RigidTransform.from_components(q_endeffector[:3], Rotation.from_euler('XYZ', q_endeffector[3:], degrees=True))
        self.q_grasps = [RigidTransform.from_matrix(grasp) for grasp in q_grasps]
        #self.q_grasp = np.argmin(np.linalg.norm(q_grasps-q_endeffector)) #closest goal pose
        # self.q_grasp = RigidTransform.from_matrix(q_grasps[0])

        # t_robot_EE
        self.trajectory = self.init_trajectory(self.q_start, RigidTransform.from_components(self.q_start.translation+[300,0,0],self.q_start.rotation)) # init trjaectory
        # t_robot_
        self.q_grasp = self.q_grasps[np.argmin([self.f_grasp(self.trajectory[-1], grasp) for grasp in self.q_grasps])]
        # self.trajectory = self.init_trajectory(self.q_start, self.q_grasp) # init trjaectory
        
        self.obj_meshes = obj_meshes
        self.w1 = 2
        self.w1_t = 1.0 #0.33
        self.w1_r = 0.0
        self.w2 = 120
        self.w3 = 0.6
        self.w3_t = 1.45#2.4
        self.w3_r = 1#1.5

        self.T_objs_robot = [RigidTransform.from_matrix(t) for t in obj_poses]
        

        # self.proximity_queries = [trimesh.proximity.ProximityQuery(mesh) for mesh in obj_meshes]

        #transform each mesh by its pose, then add to scene
        self.transformed_objs = []
        self.scene = RaycastingScene()
        
        for i in range(len(obj_meshes)):
            mesh = obj_meshes[i].transform(self.T_objs_robot[i])
            _ = self.scene.add_triangles(mesh)

        # self.obj_scenes = []
        # for mesh in obj_meshes:
        #     scene = o3d.t.geometry.RayCastingscene
        #     _ = self.scene.add_triangles(mesh)
        #     self.obj_scenes.append(scene)

    '''
    description: takes in two 6dof vectors (start and goal endeffector configs) and generates a straight trajectory between them with n steps

    input:
    q_start: 
    q_start: 
    n: int

    return:
    trajectory: ndarray
    '''
    def init_trajectory(self, q_start, q_goal, n=40):
        # assert q_start.shape[1]==6 and q_goal.shape[1]==6, "inputs aren't 6dof vectors. q_start: " + str(q_start.shape) + " q_goal: " + str(q_goal.shape)
        # print(q_start.shape)
        # print(q_start)
        # print(q_goal.shape)
        # print(q_goal)
        delta_q_xyz = q_goal.translation-q_start.translation
        # delta_q_rot = scipy.spacial.transform.sub(q_goal.rotation.as_quat(), q_start.rotation.as_quat())
        # key_rots = [q_start.rotation, q_goal.rotation]
        key_rots = Rotation.concatenate([copy(q_start.rotation), copy(q_goal.rotation)])
        key_times = [0,10]
        slerp = Slerp(key_times, key_rots)
        t_delta = 10/n
        times = [i * t_delta for i in range(n)]
        rot_interp = slerp(times)
        # delta_q_rot = q_goal.rotation.as_quat()*(q_start.rotation.as_quat().inv())
        trajectory = [None for i in range(n)]
        trajectory[0] = q_start
        trajectory[-1] = q_goal
        step_xyz = delta_q_xyz / (n-1)
        # step_rot = delta_q_rot / (n-1)

        print(delta_q_xyz)
        # print(delta_q_rot)
        for i in range(1, n-1):
            new_xyz = trajectory[i-1].translation + step_xyz
            #new_rot = q_start.rotation
            new_rot = rot_interp[i]
            trajectory[i] = scipy.spatial.transform.RigidTransform.from_components(new_xyz, new_rot)
            # trajectory[i][3:] = 179, 0, 0
            # print(trajectory[i])
        return trajectory

    '''
    part of objective function, just the euclidean distance between a pose and the goal pose
    '''
    def f_grasp(self, q1, q2):
        dist_array = np.zeros(7)
        translation_delta = q2.translation-q1.translation
        #Need to find actual quaternion difference
        # rot_delta = q2.rotation.as_quat() * (q1.rotation.as_quat()**-1)
        #rot_delta = q2.rotation.as_quat() * q1.rotation.inv().as_quat()
        dist_array[:3] = translation_delta
        #dist_array[3:] = rot_delta
        return np.linalg.norm(dist_array)
    
    
    '''
    part of obj func, queries sdf for each mesh with the xyz of the current pose
    '''
    def f_collision(self, q):
        # q_translation = [q.translation.T]

        
        # p_q_mug = p_mug_q.inv()
        # q_translation = [p_mug_q.translation.T]
        # print(q_translation.shape)
        sdf_sum = 0
        # for mesh in self.obj_meshes:
        #     value = trimesh.proximity.signed_distance(mesh,q_translation)[0]
        #     sdf_sum += value
        # for mesh, t in zip(self.proximity_queries, self.T_objs_robot):
        #     t_mesh_q = t * q
        #     q_translation = [t_mesh_q.translation.T]
        #     value = mesh.signed_distance(q_translation)[0]
        #     sdf_sum += value #**2 #min(0.0, value)** 2 # needs to be minimum as distances are negative
        sdf_sum = min(0, -self.scene.compute_signed_distance(q.translation.T))
        return sdf_sum
    
    '''
    part of obj function, not sure what this looks like yet
    From paper: 
        Measures dynamical quantites acroow the trajectory
        Example given is the integral oer squared velociy norms
    I am thinking maybe introduce an arbitrary unit of time for each step?

    -we need access to the trajectory as a whole?
    '''
    def f_smooth(self, q_next, q, q_last):
        
        dist_array = np.zeros(7)

        translation_delta = q_next.translation-(2*q.translation)+ q_last.translation

        #rot_delta = q_next.rotation.as_quat()-(q.rotation.as_quat() * q.rotation.as_quat()) * q_last.rotation.as_quat()

        dist_array[:3] = translation_delta
        #dist_array[3:] = rot_delta

        q_norm = np.linalg.norm(dist_array)

        return q_norm**2
    
    '''
    objective function, somewhat based on chomp
    '''
    def obj_func(self):
        obj_sum = 0
        for i in range(1, len(self.trajectory)-1): # for every pose in the trajectory besides the start and end
            q_next = self.trajectory[i+1]
            q = self.trajectory[i]
            q_last = self.trajectory[i-1]
            U = self.w1 * self.f_grasp(q, self.q_grasp) + self.w2 * self.f_collision(q) + self.w3 * self.f_smooth(q_next, q, q_last)
            obj_sum += U
        return obj_sum

    
    '''
    description: df_grasp/d_q
    input:
    q1
    q2
    return:
    cost
    '''
    def f_grasp_grad(self, q1, q2):
        translation_delta = q2.translation-q1.translation
        # rot_delta = q2.rotation.as_quat() * (q1.rotation.as_quat()**-1)
        #rot_delta = q2.rotation.as_quat() * (q1.rotation.inv().as_quat())


        grad_t = (translation_delta)/self.avoid_zero(np.sqrt((translation_delta)**2))
        #grad_r = (rot_delta)/self.avoid_zero(np.sqrt((rot_delta)**2))

        return grad_t#, grad_r


    '''
    description: df_collision/dq
    '''
    def f_collision_grad(self, q, eps=1e-6):
        mesh_threshold = 40
        table_threshold = 80
        # padding = 10
        q_gripper_end = RigidTransform.from_components(q.translation + [0,0,-15], q.rotation)
        

        sdf_grad_sum = np.zeros((1,3))

        

        # for mesh in self.obj_meshes:
        #     q_xyz = [p_mug_q.translation.T]  
        #     value = trimesh.proximity.signed_distance(mesh,q_xyz)[0]
        #     closest_point = trimesh.proximity.closest_point(mesh, q_xyz)[0]
            
            
        #     # No collision
        #     if value >= 0:
        #         direction = q_xyz[0] - closest_point
        #         norm = np.linalg.norm(direction)

        #         if norm > eps:
        #             sdf_grad = direction / norm

        #             # cost = value**2
        #             # grad = 2 * value * grad_value
        #             sdf_grad_sum += 2 * value * sdf_grad
        #     else:
        #         sdf_grad_sum += 2 * value
        # for mesh, t in zip(self.proximity_queries,self.T_objs_robot):
        #     T_mesh_q = t * q_gripper_end
        #     q_xyz = [T_mesh_q.translation.T]
        #     value = mesh.signed_distance(q_xyz)[0]
        #     # print(value)
        #     closest_point = mesh.on_surface(q_xyz)[0]
        #     dist_grad = (closest_point-q_xyz)/self.avoid_zero(np.sqrt((closest_point-q_xyz)**2))
        #     if value > -mesh_threshold:
        #         sdf_grad_sum += dist_grad
        
        #scene not including table
        q_xyz = q.translation.T
        value = -self.scene.compute_signed_distance(q_xyz)
        closest_point = self.scene.compute_closest_point(q_xyz).points
        dist_grad = (closest_point-q_xyz)/self.avoid_zero(np.sqrt((closest_point-q_xyz)**2))
        if value > -mesh_threshold:
            sdf_grad_sum += dist_grad

        #table collision avoidance
        value = q.translation[2]
        closest_point = q.translation
        closest_point[2] = 0.1
        dist_grad = (closest_point-q.translation)/self.avoid_zero(np.sqrt((closest_point-q.translation)**2))
        if value < table_threshold:
            sdf_grad_sum -= dist_grad
        # if np.any([x!=0 for x in sdf_grad_sum]):
        #     print("sum")
        #     print(sdf_grad_sum)
        return sdf_grad_sum


    '''
    description: q_norm = np.linalg.norm(q_next - 2*q + q_last), dq_norm/dq
    input:
    q_next
    q
    q_last
    '''
    def f_smooth_grad(self, q_next, q, q_last):
        # q = self.trajectory[q_idx]
        # q_last = self.trajectory[q_idx-1]
        # q_next = self.trajectory[q_idx+1]

        # return -4 *(q_next - 2*q + q_last)
        grad_t = -4 *(q_next.translation - 2*q.translation + q_last.translation)
        #Need to change to proper quaternion difference and sum
        # grad_r = -4 *((q.rotation * q.rotation) * q_next.rotation.inv() * q_last.rotation)

        return grad_t#, grad_r



    '''
    description: objective function U = w1*f_grasp + w2*f_collision + w3*f_smooth, so partial derivate wrt q will be sum of each function grads multiplied by weights.
    
    what this should do:
    get partial derivatives for cost function components
    get partial derivatives for weigths
    re-select best goal pose

    
    partial derivatives
    U=w1*F_grasp + w2*F_collision + w3*F_smooth
    params = [q, q_goal, w1, w2, w3]
    dU/dq = w1*dF_grasp/d_q + w2*dF_collision/d_q + w3 * df_smooth/d_q
    dU/dW1 = F_grasp
    dU/dw2 = F_collision
    dU/dw3 = F_smooth
    q_goal is discrete, so:
    q_goal = closest candidate pose to q, check at end

    '''
    def optimize_trajectory(self):

        #do sgd until iteration limit or objective cost below some threshold
        iteration_limit = 1000 #2000
        # lr = 1e-1 #1e-2
        threshold = 1
        U_last = 0
        for it in range(iteration_limit):
            lr = (8e-2)*(1-(it/(1.2*iteration_limit))) # decreasing learning rate
            for i in range(1, len(self.trajectory)):
            # first get partial derivatives for weights
                q = self.trajectory[i]
                q_goal_new = None
                q_last = self.trajectory[i-1]

                if i < (len(self.trajectory)-1):
                    q_next = self.trajectory[i+1]
                else:
                    q_next = self.q_grasp         
                # delta_w1 = lr * self.f_grasp(q, self.q_grasp)
                # delta_w2 = lr * self.f_collision(q)
                # delta_w3 = lr * self.f_smooth(q_next, q, q_last)

                # if i % 1 == 0: # change this in case recalculating the goal every time is too much
                #     q_goal_new = self.q_grasps[np.argmin(np.linalg.norm(self.q_grasps-q, axis=-1))]

                # if q_goal_new is not None:
                #     self.q_grasp = q_goal_new
                
                # self.w1 -= delta_w1
                # self.w2 -= delta_w2
                # self.w3 -= delta_w3
                # delta_q = lr * (self.w1 * self.f_grasp_grad(q, self.q_grasp) + self.w2 * self.f_collision_grad(q) + self.w3 * self.f_smooth_grad(q, q_next, q, q_last))
                
                grasp_grad_t = self.f_grasp_grad(q, self.q_grasp)
                smooth_grad_t = self.f_smooth_grad(q_next, q, q_last)

                delta_q_t = lr * (self.w1_t * grasp_grad_t + self.w2 * self.f_collision_grad(q) - self.w3_t * smooth_grad_t)
                # delta_q_r = lr * (self.w1_r * grasp_grad_r - self.w3_r * smooth_grad_r)

                # if delta_q_r.any(None):
                #     print (delta_q_r)
                # if delta_q_t.any(None): 
                #     print (delta_q_t)

                #if(delta_q_r.all(0)):
                #    new_pose = RigidTransform.from_components(self.trajectory[i].translation + delta_q_t.flatten(), self.trajectory[i].rotation)
                #else:

                # new_pose = RigidTransform.from_components(self.trajectory[i].translation + delta_q_t.flatten(), Rotation.from_quat(self.avoid_zero(self.trajectory[i].rotation.as_quat()+delta_q_r.flatten())))
                new_pose = RigidTransform.from_components(self.trajectory[i].translation + delta_q_t.flatten(), self.trajectory[i].rotation)
                
                self.trajectory[i] = new_pose

                # self.trajectory[i].translation -= delta_q_t.flatten()
                # self.trajectory[i].translation -= delta_q_r.flatten()

                # Project endpoint to closest grasp in grasp set
                # q_end = self.q_grasp.copy()
                # self.q_grasp = self.q_grasps[np.argmin(np.linalg.norm(self.q_grasps-q_end, axis=-1))]

                # # Propagate endpoint correction back through trajectory
                # delta_end = self.q_grasp - q_end
                # pts = len(self.trajectory)
                # for i in range(1, pts):
                #     alpha = i/pts
                #     self.trajectory[i] += alpha * delta_end 


            # (Optional) joint limit 
            # if hasattr(self, "q_min") and hasattr(self, "q_max"):
            #     self.trajectory = np.clip(
            #         self.trajectory,
            #         self.q_min,
            #         self.q_max
            #     )
                

            if it % 25 == 0: # only check every 25 iterations 
                
                U = self.obj_func()
                print("--------------------------")
                print("iteration:")
                print(it)
                print("Current obj val: ", U)
                print("weights:")
                print(self.w1)
                print(self.w2)
                print(self.w3)
                # if np.abs(U-U_last) < 100 or U-U_last > 500:
                #     return
                U_last = U

                # print(self.q_grasp)
                # print(self.q_grasps)
                self.q_grasp = self.q_grasps[np.argmin([self.f_grasp(self.trajectory[-1], grasp) for grasp in self.q_grasps])]

    def get_euler_trajectory(self):
        viz_trajectory = self.trajectory.copy()
        viz_trajectory.append(self.q_grasp) # make last point the best grasp pose
        euler_trajectory = np.ndarray((len(viz_trajectory),6))
        # self.q_grasp = RigidTransform.from_matrix(self.q_grasps[0])
        for i in range(len(viz_trajectory)):
            euler_transform = np.ndarray((1,6))
            euler_transform[0,:3] = np.asarray([viz_trajectory[i].translation])
            euler_transform[0,3:] = np.asarray([viz_trajectory[i].rotation.as_euler('xyz', degrees=True)])
            euler_trajectory[i] = euler_transform
        return euler_trajectory
    
    '''
    avoid using zero as denominator
    '''
    def avoid_zero(self, x):
        eps = np.finfo(x.dtype).eps
        return np.maximum(x, eps)
    
    def get_finger_points(self, q, offset=0.013):
        left_gripper  = q[:3, :3] @ np.array([0,  offset, 0]) + q[:3, 3]
        right_gripper = q[:3, :3] @ np.array([0, -offset, 0]) + q[:3, 3]
        return left_gripper, right_gripper

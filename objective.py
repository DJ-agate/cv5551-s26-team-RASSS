#based on CHOMP and NGDF

import numpy as np
import trimesh
import open3d as o3d
import scipy

from scipy.spatial.transform import RigidTransform, Rotation

from utils.math_utils import matrix_to_pose

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
    def __init__(self, q_endeffector, q_grasps, obj_meshes):
        self.q_start = RigidTransform.from_components(q_endeffector[:3], Rotation.from_euler('XYZ', q_endeffector[3:], degrees=True))
        self.q_grasps = [([RigidTransform.from_matrix(grasp) for grasp in q_grasps])]
        #self.q_grasp = np.argmin(np.linalg.norm(q_grasps-q_endeffector)) #closest goal pose
        self.q_grasp = RigidTransform.from_matrix(q_grasps[0])
        self.trajectory = self.init_trajectory(self.q_start, self.q_grasp) # init trjaectory
        self.obj_meshes = obj_meshes
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1

        self.traj_smooth_hist = []

    '''
    description: takes in two 6dof vectors (start and goal endeffector configs) and generates a straight trajectory between them with n steps

    input:
    q_start: 
    q_start: 
    n: int

    return:
    trajectory: ndarray
    '''
    def init_trajectory(self, q_start, q_goal, n=10):
        # assert q_start.shape[1]==6 and q_goal.shape[1]==6, "inputs aren't 6dof vectors. q_start: " + str(q_start.shape) + " q_goal: " + str(q_goal.shape)
        # print(q_start.shape)
        # print(q_start)
        # print(q_goal.shape)
        # print(q_goal)
        delta_q_xyz = q_goal.translation-q_start.translation
        delta_q_rot = q_goal.rotation.as_quat()-q_start.rotation.as_quat()
        trajectory = [None for i in range(n)]
        trajectory[0] = q_start
        trajectory[-1] = q_goal
        step_xyz = delta_q_xyz / (n-1)
        step_rot = delta_q_rot / (n-1)
        print(delta_q_xyz)
        print(delta_q_rot)
        for i in range(1, n-1):
            new_xyz = trajectory[i-1].translation + step_xyz
            #new_rot = q_start.rotation
            new_rot = Rotation.from_quat(trajectory[i-1].rotation.as_quat() + step_rot)
            trajectory[i] = scipy.spatial.transform.RigidTransform.from_components(new_xyz, new_rot)
            # trajectory[i][3:] = 179, 0, 0
            print(trajectory[i])
        return trajectory

    '''
    part of objective function, just the euclidean distance between a pose and the goal pose
    '''
    def f_grasp(self, q1, q2):
        return np.linalg.norm(q2-q1)
    
    '''
    part of obj func, queries sdf for each mesh with the xyz of the current pose
    '''
    def f_collision(self, q):
        sdf_sum = 0
        for mesh in self.obj_meshes:
            q_xyz = q[:3]  
            value = trimesh.proximity.signed_distance(mesh,q_xyz)[0]
            sdf_sum += value
        return sdf_sum
    
    '''
    part of obj function, not sure what this looks like yet
    From paper: 
        Measures dynamical quantites acroow the trajectory
        Example given is the integral oer squared velociy norms
    I am thinking maybe introduce an arbitrary unit of time for each step?

    -we need access to the trajectory as a whole?
    '''
    def f_smooth(self, q_idx):
        q = self.trajectory[q_idx]
        q_last = self.trajectory[q_idx-1]
        q_next = self.trajectory[q_idx+1]
        
        #dist_last = np.linalg.norm(q-q_last)
        #dist_next = np.linalg.norm(q_next-q)
        q_norm = np.linalg.norm(q_next - 2*q + q_last)

        return q_norm**2

    '''
    objective function, somewhat based on chomp
    '''
    def obj_func(self):
        obj_sum = 0
        for i in range(1, self.trajectory.shape[1]): # for every pose in the trajectory besides the start and end
            q = self.trajectory[i]
            U = self.w1 * self.f_grasp(q, self.q_grasp) + self.w2 * self.f_collision(q) + self.w3 * self.f_smooth(q)
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
        return (q2-q1)/np.sqrt((q2-q1)^2)

    '''
    description: df_collision/dq
    '''
    def f_collision_grad(self, q):
        sdf_grad_sum = 0
        for mesh in self.obj_meshes:
            q_xyz = q[:3]  
            value = trimesh.proximity.signed_distance(mesh,q_xyz)[0]
            closest_point = trimesh.proximity.closest_point(mesh, q_xyz)
            dist_grad = (closest_point-q_xyz)/np.sqrt((closest_point-q_xyz)^2)
            if value > 0:
                sdf_grad_sum += dist_grad
            else:
                sdf_grad_sum -= dist_grad
                
        return sdf_grad_sum

    '''
    idk bro
    '''
    def f_smooth_grad(self, q):
        pass

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
        iteration_limit = 1000
        lr = 10e-4
        threshold = 1
        for it in range(iteration_limit):
            for i in range(1, self.trajectory.shape[1]):
            # first get partial derivatives for weights
                q = self.trajectory[i]
                q_goal_new = None
                q_last = self.trajectory[i-1]
                delta_w1 = lr * self.f_grasp(q, self.q_grasp)
                delta_w2 = lr * self.f_collision(q)
                delta_w3 = lr * self.f_smooth(q, q_last)
                delta_q = lr * (self.w1 * self.f_grasp_grad(q, self.q_grasp) + self.w2 * self.f_collision_grad(q) + self.w3 * self.f_smooth_grad(q, q_last))
                if i % 1 == 0: # change this in case recalculating the goal every time is too much
                    q_goal_new = self.q_grasps[np.argmin(np.linalg.norm(self.q_grasps-q))]
                self.w1 -= delta_w1
                self.w2 -= delta_w2
                self.w3 -= delta_w3
                self.trajectory[i] += delta_q
                if q_goal_new is not None:
                    self.q_grasp = q_goal_new
                
            if it % 25 == 0: # only check every 25 iterations 
                U = self.obj_func()
                print("Current obj val: ", U)
                if U < threshold:
                    return

    def get_euler_trajectory(self):
        euler_trajectory = np.ndarray((len(self.trajectory),6))
        for i in range(len(self.trajectory)):
            euler_transform = np.ndarray((1,6))
            euler_transform[0,:3] = np.asarray([self.trajectory[i].translation])
            euler_transform[0,3:] = np.asarray([self.trajectory[i].rotation.as_euler('XYZ', degrees=True)])
            euler_trajectory[i] = euler_transform
        return euler_trajectory
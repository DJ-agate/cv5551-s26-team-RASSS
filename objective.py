#based on CHOMP and NGDF

import numpy as np
import trimesh

'''
Description: Optimizes a trajectory

'''
class objective_optimizer:
    '''
    inputs:
    q_endeffector: initial position of endeffector, 6x1 vector
    q_grasps: ndarray of candidate grasps
    obj_meshes: list of object meshes
    '''    
    def __init__(self, q_endeffector, q_grasps, obj_meshes):
        self.q_grasp = np.argmin(np.linalg.norm(q_grasps-q_endeffector)) #closest goal pose
        self.trajectory = self.init_trajectory(q_endeffector, self.q_grasp) # init trjaectory
        self.obj_meshes = obj_meshes
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1

    '''
    description: takes in two 6dof vectors (start and goal endeffector configs) and generates a straight trajectory between them with n steps

    input:
    q_start: 6x1 ndarray 
    q_start: 6x1 ndarray
    n: int

    return:
    trajectory: 6xn ndarray
    '''
    def init_trajectory(self, q_start, q_goal, n):
        assert q_start.shape[0]==6 and q_goal.shape[0]==6, "inputs aren't 6dof vectors"
        delta_q = q_goal-q_start
        trajectory = q_start
        for i in range(n):
            q_i = trajectory[i-1] + (delta_q/n)
            trajectory.concatenate(q_i, axis=1)
        trajectory.concatenate(q_goal)

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
    '''
    def f_smooth(self, q):
        pass

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
                #do x
                pass
            else:
                # do y
                pass
        return sdf_grad_sum

    '''
    idk bro
    '''
    def f_smooth_grad(self, q):
        #todo
        pass

    '''
    description: objective function U = w1*f_grasp + w2*f_collision + w3*f_smooth, so partial derivate wrt q will be sum of each function grads multiplied by weights.
    
    what this should do:
    get partial derivatives for cost function components
    get partial derivatives for weigths
    re-select best goal pose


    '''
    def optimize_trajectory(self):

        #do sgd until iteration limit or objective cost below some threshold
        iteration_limit = 1000
        for it in range(iteration_limit):
            for i in range(1, self.trajectory.shape[1]):
            # first get partial derivatives for 
                pass
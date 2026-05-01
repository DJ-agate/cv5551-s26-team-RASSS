import trimesh
import numpy as np


class Planner():
    def __init__(self):
        self.env = None
        self.traj = None
        self.cost = None

    def grasp_init(self,env):
        self.traj.end = None #t_robot_mug_pose

    def plan(self, trajectory):


        for t in range(100):

            grad = get_grad(cur_vec, init_vec, d, eps=1e-6, t_robot_model=t_robot_model)
            # next_vec = optimizer.step(cur_vec, grad)
            next_vec = cur_vec - alpha * grad
            cur_cost = total_cost(next_vec, init_vec, d, t_robot_model=t_robot_model)
            if abs(prev_cost - cur_cost) < 1e-4:
                print(f"Converged at iter {t}")
                break
            prev_cost = cur_cost
            cur_vec = next_vec

            tx, ty, tz, tr, tp, tyaw = cur_vec
            print(cur_vec)
            # Use mm as unit of robot motion
            #arm.set_position(tx*1000, ty*1000, tz*1000, tr, tp, tyaw, is_radian=True)

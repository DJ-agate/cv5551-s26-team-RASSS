import pickle
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation
import numpy as np
from xarm.wrapper import XArmAPI



robot_ip = '192.168.1.172'


#import get_mug_transform
T_cam_mug = get_mug_from_april() #todo once complete
#T_cam_mug = get_mug_transform() todo
T_cam_robot = get_transform_camera_robot()


# Initialize Lite6 Robot
arm = XArmAPI(robot_ip)
arm.connect()
arm.motion_enable(enable=True)

arm.set_mode(0)
arm.set_state(0)
arm.move_gohome(wait=True)

grasp_poses = []
usr_input = input("Press c to capture a pose, d to delete the last one, and q to quit")
while usr_input != 'q':
    if usr_input == 'c':
        #T_robot_pose
        grasp_pose = arm.get_position()
        #T_pose_robot
        grasp_pose = np.linalg.inv(grasp_pose)
        #T_pose_cam
        grasp_pose = np.linalg.inv(T_cam_robot) @ grasp_pose
        #T_cam_pose
        grasp_pose = np.linalg.inv(grasp_pose)
        grasp_poses.append(grasp_pose)
    if usr_input == 'd':
        grasp_poses.pop()
    usr_input = input("Press c to capture a pose, d to delete the last one, and q to quit")

with open('./poses.pkl', 'wb') as fp:
    pickle.dump(grasp_poses, fp)
arm.disconnect()

import pickle

from utils.zed_camera import ZedCamera
import cv2, time


from utils.vis_utils import draw_grasp_poses
from utils.vis_utils import plot_trajectory
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation as R
import numpy as np
from xarm.wrapper import XArmAPI
from objective import objective_optimizer
import trimesh

#for robot frame
TAG_SIZE = 0.08
GRIPPER_LENGTH = 0.069 * 1000
TCP_OFFSET = [0,0,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205

robot_ip = '192.168.1.172'


def main():
    mug_poses = np.load("poses.pkl", allow_pickle=True)
    print("mug poses pkl:")
    print(mug_poses[0])
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset(TCP_OFFSET)
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(2.5)

    try:
        # Get Observation
        cv_image = zed.image

        
        cv_image = zed.image

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        # if t_cam_robot is None:
        #     return
        t_cam_tag = None
        t_robot_cube , t_cam_tag, tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot))
        #print(t_cam_tag)
        #import get_mug_transform

        t_cam_mug = get_mug_from_april(t_cam_tag, tag_id) #todo once complete
        
        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug

        t_robot_mug[:3, 3] = t_robot_mug[:3, 3] *1000
        
        #print("t_robot_mug")
        #print(t_robot_mug)
        '''
        NEEDS TO BE EDIT SAFE! USE .COPY()
        '''
        draw_grasp_poses(cv_image, camera_intrinsic, np.copy(mug_poses), t_cam_mug )

        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()

        ##TODO: IMPORTANT to transform all mug poses before passing them in
        
        print("t_mug_grasp")
        print(mug_poses[0])
        grasp_pose = t_robot_mug @ mug_poses[0]
        print("t_robot_mug_grasp = ", grasp_pose)

        mesh = trimesh.load_mesh("Mug_w_tags.stl")
        mesh.apply_scale(0.02)
        obj_opt = objective_optimizer(arm.get_position()[1],[grasp_pose],[mesh])
        print("init EE pose: ")
        print(arm.get_position()[1])
        
        plot_trajectory(obj_opt.trajectory)
        #arm.move_arc_lines(obj_opt.trajectory)
        #print(obj_opt.trajectory)
        
        # obj_opt.optimize_trajectory


        # Grasp
        # x = grasp_pose[0][3]
        # y = grasp_pose[1][3]
        # z = grasp_pose[2][3]
        
        # rot_pose = np.eye(3)
        # rot_pose[:3][:3] = grasp_pose[0:3,0:3]
        # rot = R.from_matrix(rot_pose)
        # angles = rot.as_euler("xyz",degrees=True)
        # roll = angles[0]
        # pitch = angles[1]
        # yaw = angles[2]
        # arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True)

        for i in range(len(obj_opt.trajectory)):
            grasp_pose = obj_opt.trajectory[i]
            x = grasp_pose[0]
            y = grasp_pose[1]
            z = grasp_pose[2]
            
            # rot_pose = np.eye(3)
            # rot_pose[:3][:3] = grasp_pose[0:3,0:3]
            # rot = R.from_matrix(rot_pose)
            
            roll = grasp_pose[3]
            pitch = grasp_pose[4]
            yaw = grasp_pose[5]
                
            
            arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True)
            time.sleep(1)
        #arm.set_position(x,y,z+40,roll,pitch,yaw,is_radian=None,wait=True)
        #arm.open_lite6_gripper()
        time.sleep(0.5)
        # arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True, speed=200)


        
    
    finally:
        # Close Lite6 Robot
        #arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
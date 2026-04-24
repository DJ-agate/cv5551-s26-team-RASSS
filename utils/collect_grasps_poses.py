import pickle

from utils.zed_camera import ZedCamera
import cv2, time


from utils.vis_utils import draw_pose_axes
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation as R
import numpy as np
from xarm.wrapper import XArmAPI

TAG_SIZE = 0.08
# GRIPPER_LENGTH = 0.069 * 1000
# TCP_OFFSET = [0,-2,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205

GRIPPER_LENGTH = 0.069 * 1000
TCP_OFFSET = [0,0,GRIPPER_LENGTH,0,0,0]


robot_ip = '192.168.1.172'

def main():

    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    try:
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

        print(t_robot_mug)
        t_robot_mug[:3, 3] = t_robot_mug[:3, 3] *1000

        print(t_robot_mug)

        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        #draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_mug)
        #draw_pose_axes(cv_image, camera_intrinsic, (t_cam_mug @ numpy.linalg.inv(y_square)))
        #draw_pose_axes(cv_image, camera_intrinsic, (t_cam_mug @ y_square))
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()



        # Initialize Lite6 Robot
        arm = XArmAPI(robot_ip)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_tcp_offset(TCP_OFFSET)
        arm.set_mode(0)
        arm.set_state(0)
        arm.move_gohome(wait=True)

        grasp_poses = []

        grasp_xyz_rot = arm.get_position()
        grasp_xyz = grasp_xyz_rot[1][:3]
        grasp_rpy = grasp_xyz_rot[1][3:]
        r = R.from_euler('xyz', grasp_rpy, degrees=True)
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = r.as_matrix()
        grasp_pose[:3, 3] = grasp_xyz
        print("starting pose")
        print(grasp_pose)
        usr_input = input("Press c to capture a pose, d to delete the last one, and q to quit \n")
        while usr_input != 'q':
            if usr_input == 'c':
                #T_robot_pose
                grasp_xyz_rot = arm.get_position()
                grasp_xyz = grasp_xyz_rot[1][:3]
                grasp_rpy = grasp_xyz_rot[1][3:]
                r = R.from_euler('xyz', grasp_rpy, degrees=True)
                T_robot_pose = np.eye(4)
                T_robot_pose[:3, :3] = r.as_matrix()
                T_robot_pose[:3, 3] = grasp_xyz

                print("T_robot_pose: ")
                print(T_robot_pose) #T_robot_pose

                #T_pose_robot = np.linalg.inv(T_robot_pose)
                #T_pose_mug = T_pose_robot @ t_robot_mug
                #T_mug_pose = np.linalg.inv(T_pose_mug)

                T_mug_pose = np.linalg.inv(t_robot_mug) @ T_robot_pose 
                
                print("T_mug_pose: ")
                print(T_mug_pose)

                grasp_poses.append(T_mug_pose)
            if usr_input == 'd':
                grasp_poses.pop()
            usr_input = input("Press c to capture a pose, d to delete the last one, and q to quit \n")

        with open('./poses.pkl', 'wb') as fp:
            pickle.dump(grasp_poses, fp)
        arm.disconnect()

    finally:
        # Close Lite6 Robot
        time.sleep(0.5)

        # Close ZED Camera
        zed.close()
if __name__ == "__main__":
    main()
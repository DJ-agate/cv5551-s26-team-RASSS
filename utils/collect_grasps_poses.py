import pickle

from utils.zed_camera import ZedCamera
import cv2, time


from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation as R
import numpy as np
from xarm.wrapper import XArmAPI


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
        print(t_cam_tag)
        #import get_mug_transform

        t_cam_mug = get_mug_from_april(t_cam_tag, tag_id) #todo once complete
        
        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug



        # Initialize Lite6 Robot
        arm = XArmAPI(robot_ip)
        arm.connect()
        arm.motion_enable(enable=True)

        arm.set_mode(0)
        arm.set_state(0)
        arm.move_gohome(wait=True)

        grasp_poses = []
        usr_input = input("Press c to capture a pose, d to delete the last one, and q to quit \n")
        while usr_input != 'q':
            if usr_input == 'c':
                #T_robot_pose
                grasp_xyz_rot = arm.get_position()
                grasp_xyz = grasp_xyz_rot[1][:3]
                grasp_rpy = grasp_xyz_rot[1][3:]
                r = R.from_euler('xyz', grasp_rpy, degrees=True)
                grasp_pose = np.eye(4)
                grasp_pose[:3, :3] = r.as_matrix()
                grasp_pose[:3, 3] = grasp_xyz
                print(grasp_pose) #T_robot_pose

                T_pose_robot = np.linalg.inv(grasp_pose)
                T_pose_mug = T_pose_robot @ t_robot_mug
                T_mug_pose = np.linalg.inv(T_pose_mug)

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
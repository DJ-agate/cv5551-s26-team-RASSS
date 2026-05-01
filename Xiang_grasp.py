
from xarm.wrapper import XArmAPI

import cv2, time

import numpy as np

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation

import open3d as o3d

from Xiang_trajectory import grasp_with_sdf

INIT_POSE = np.array([[ 9.99999999e-01,  3.09930336e-05,  2.70048509e-05,  0.0869530030],
                    [ 3.09970475e-05, -9.99999988e-01, -1.48648855e-04,  0.00829000000],
                    [ 2.70002435e-05,  1.48649692e-04, -9.99999989e-01,  0.154287003],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])



def main():

     # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic


    try:
        cv_image = zed.image

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return


        t_cam_tag = None
        t_robot_cube , t_cam_tag, tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot))
        #print(t_cam_tag)

        t_cam_mug = get_mug_from_april(t_cam_tag, tag_id)
        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug
        print("t_robot_mug")
        print(t_robot_mug)
        grasp_with_sdf(INIT_POSE, d=0.005, t_robot_model=t_robot_mug)
    
    finally:
        # Close Lite6 Robot
        time.sleep(0.5)

        # Close ZED Camera
        zed.close()





if __name__ == "__main__":
    main()

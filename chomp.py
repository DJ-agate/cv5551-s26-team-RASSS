import cv2, time

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april

from objective import objective_optimizer

import numpy as np
import trimesh

TAG_SIZE = 0.08
# GRIPPER_LENGTH = 0.069 * 1000
# TCP_OFFSET = [0,-2,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205


INIT_POSE = np.array([[ 9.99999999e-01,  3.09930336e-05,  2.70048509e-05,  0.0869530030],
                    [ 3.09970475e-05, -9.99999988e-01, -1.48648855e-04,  0.00829000000],
                    [ 2.70002435e-05,  1.48649692e-04, -9.99999989e-01,  0.154287003],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])



def main():
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic


    try:
        # Get Observation
        cv_image = zed.image

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return


        t_cam_tag = None
        t_robot_cube , t_cam_tag, tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot))
        print(t_cam_tag)

        t_cam_mug = get_mug_from_april(t_cam_tag, tag_id)

        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug


        #THESE OFFSETS WILL CHANGE, NEED TO MEASURE FINAL OFFSETS WHEN TAGS ARE GLUED TO CUP

        # Visualization
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

        mesh = trimesh.load_mesh("Mug_w_tags.stl")
        mesh.apply_scale(0.02)
        obj_opt = objective_optimizer(INIT_POSE,[t_robot_mug],[mesh])
        obj_opt.optimize_trajectory


    finally:
        # Close Lite6 Robot
        time.sleep(0.5)

        # Close ZED Camera
        zed.close()
    

    pass
if __name__ == "__main__":
    main()
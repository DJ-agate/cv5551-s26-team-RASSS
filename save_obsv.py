
import cv2, numpy, time


from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation

import open3d as o3d

TAG_SIZE = 0.08
# GRIPPER_LENGTH = 0.069 * 1000
# TCP_OFFSET = [0,-2,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205



def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic


    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud
        cv2.imwrite("img_table_mug.png", cv_image)
        o3d.io.write_point_cloud("pcd_table_mug.pcd", point_cloud)


        

    finally:
        # Close Lite6 Robot
        time.sleep(0.5)

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
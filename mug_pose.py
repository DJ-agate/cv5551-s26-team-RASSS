
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

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return


        t_cam_cube = None
        t_robot_cube , t_cam_cube = get_transform_cube(cv_image, camera_intrinsic, numpy.linalg.inv(t_cam_robot))
        print(t_cam_cube)

        t_cam_mug = get_mug_from_april(t_cam_cube, 5)

        x = 0   #Meters offset to center
        y = 0.042
        z = 0
        r = Rotation.from_euler('zyx', [0,0,90], degrees=True)
        y_square = numpy.eye(4)
        y_square[:3,:3] = r.as_matrix()
        y_square[:3, 3] = [x,y,z]



        #THESE OFFSETS WILL CHANGE, NEED TO MEASURE FINAL OFFSETS WHEN TAGS ARE GLUED TO CUP

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        #draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        #draw_pose_axes(cv_image, camera_intrinsic, t_cam_mug)
        #draw_pose_axes(cv_image, camera_intrinsic, (t_cam_mug @ numpy.linalg.inv(y_square)))
        draw_pose_axes(cv_image, camera_intrinsic, (t_cam_mug @ y_square))
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()
        

    finally:
        # Close Lite6 Robot
        time.sleep(0.5)

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
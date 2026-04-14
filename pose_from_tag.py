import cv2, numpy, time
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot


#for robot frame
TAG_SIZE = 0.08
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function uses visual fiducial detection to find the cube's pose in the camera's view, 
    then transforms that pose into the robot's global coordinate system. 

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no cube tag is detected, returns None.
    """

    # camera_pose = T_cam_robot
    # TODO
    # Initialize AprilTag Detector
    detector = Detector(families='tag36h11')
    if len(observation.shape) > 2:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)

    tags = detector.detect(observation, estimate_tag_pose=True, tag_size=CUBE_TAG_SIZE, camera_params=[camera_intrinsic[0][0], camera_intrinsic[1][1], camera_intrinsic[0][2], camera_intrinsic[1][2]])
    tags = [tags[i] for i in range(len(tags)) if tags[i].tag_id > 3]

    print(f'Number of tags found: {len(tags)}')
    # print(tags[0])

    if len(tags) == 0:
        return None

    cube_pose = numpy.eye(4)
    cube_pose[:3, :3] = tags[0].pose_R
    cube_pose[:3, 3] = tags[0].pose_t.flatten()



    print(tags[0].pose_t.flatten())
    print(cube_pose)


    return (camera_pose @ cube_pose, cube_pose)
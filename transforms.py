import cv2, numpy
from pupil_apriltags import Detector

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

from scipy.spatial.transform import Rotation
import open3d as o3d

TAG_SIZE = 0.08

# top-left, top-right, bottom-left, bottom-right
TAG_CENTER_COORDINATES = [[0.38, 0.4],
                         [0.38, -0.4],
                         [0.0, 0.4],
                         [0.0, -0.4]]
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205


def get_mug_from_april(tag_transform, tag_num):
    print("TAG NUMBER: ", tag_num)
    # 4 cases, each for the april tags on a visible surface of the mug
    if tag_num == 6: # LEFT OF HANDLE
        x = 0  #Meters offset to center
        y = 0 #-0.042
        z = 0.0425
        r = Rotation.from_euler('xyz', [-90,-108,0], degrees=True) # ORIENTATION CORRECT
        r = r.as_matrix()

        offset = numpy.eye(4)
        offset[:3,:3] = r
        offset[:3, 3] = [x,y,z]
        
        mug_pose = tag_transform @ offset
    elif tag_num == 7: # CENTER OPPOSITE OF HANDLE
        x = 0   #Meters offset to center
        y = 0.0425
        z = 0
        r = Rotation.from_euler('zyx', [0,0,90], degrees=True) # ORIENTATION CORRECT
        y_square = numpy.eye(4)
        y_square[:3,:3] = r.as_matrix()
        y_square[:3, 3] = [x,y,z]

        
        mug_pose = tag_transform @ numpy.linalg.inv(y_square)
    elif tag_num == 8: #RIGHT OF HANDLE
        x = 0.0   
        y = 0.0
        z = 0.0425
        r = Rotation.from_euler('zyx', [108,0,-90], degrees=True) # ORIENTATION CORRECT
        r = r.as_matrix()
        offset = numpy.eye(4)
        offset[:3,:3] = r
        offset[:3, 3] = [x,y,z]
        mug_pose = tag_transform @ offset
    elif tag_num == 9: # BOTTOM of MUG
        x = 0.0425
        y = 0
        z = 0
        r = Rotation.from_euler('zyx', [0,0,90], degrees=True) # ORIENTATION CORRECT
        r = r.as_matrix()
        offset = numpy.eye(4)
        offset[:3,:3] = r
        offset[:3, 3] = [x,y,z]
        mug_pose = tag_transform @ offset
    else:
        print("NONE OF THESE TAGS ARE CORRECT")
        
        mug_pose = tag_transform @ offset


    return mug_pose

def get_transform_cube(observation, camera_intrinsic, camera_pose, tag_range=[4,9]):
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
    tags = [tags[i] for i in range(len(tags)) if (tags[i].tag_id >= tag_range[0]) and (tags[i].tag_id <= tag_range[1])]

    print(f'Number of tags found: {len(tags)}')
    # print(tags[0])

    if len(tags) == 0:
        return None

    # Get Transformation
    # print(tags)
    # print(tags[0].tag_id)
    # print(tags[0].pose_R)
    # print(tags[0].pose_t)
    cube_pose = numpy.eye(4)
    cube_pose[:3, :3] = tags[0].pose_R
    
    cube_pose[:3, 3] = tags[0].pose_t.flatten()



    #print(tags[0].pose_t.flatten())
    #print(cube_pose)
    # success, rotation_vec, translation = cv2.solvePnP(tags[0].pose_t.flatten(), tags[0].center, camera_intrinsic, None)
    # if success is not True:
    #     print('PnP Calculation Failed.')
    #     return None
    # rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    # transform_mat = numpy.eye(4)
    # transform_mat[:3, :3] = rotation_mat
    # transform_mat[:3, 3] = translation.flatten()
    # transform_mat = []

    return (camera_pose @ cube_pose, cube_pose, tags[0].tag_id)

def get_pnp_pairs(tags):
    """
    Extract corresponding 3D world coordinates and 2D image coordinates for 
    the corners of detected AprilTags.

    This function iterates through the detected tags, filters for specific tag IDs 
    (0 through 3), and computes the 3D world coordinates of their four corners 
    based on predefined center coordinates and tag size. It maps these to the 
    corresponding 2D pixel coordinates found in the image.

    Parameters
    ----------
    tags : list
        A list of AprilTag detection objects returned by the pupil_apriltags detector.

    Returns
    -------
    world_points : numpy.ndarray
        An (N, 3) array of 3D world coordinates for the tag corners.
    image_points : numpy.ndarray
        An (N, 2) array of corresponding 2D image pixel coordinates for the tag corners.
    """
    world_points = numpy.empty([0, 3])
    image_points = numpy.empty([0, 2])

    for tag in tags:
        
        if tag.tag_id > 3:
            continue
        
        tag_center = TAG_CENTER_COORDINATES[tag.tag_id]

        # Bottom left corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[0]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Bottom right corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[1]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Top right corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[2]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Top left corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[3]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

    return world_points, image_points

def get_transform_camera_robot(observation, camera_intrinsic):
    """
    Calculate the 4x4 transformation matrix from the camera frame to the 
    robot base frame using AprilTag detections.

    The function detects AprilTags in the provided image, retrieves 
    the 3D-2D point correspondences, and uses the Perspective-n-Point (PnP) algorithm 
    to estimate the pose of the camera.

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.

    Returns
    -------
    transform_mat : numpy.ndarray or None
        A 4x4 transformation matrix representing the rotation and translation,
        or None if insufficient valid tags are found or the PnP calculation fails.
    """

    # Initialize AprilTag Detector
    detector = Detector(families='tag36h11')

    # Detect AprilTag Points
    if len(observation.shape) > 2:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    tags = detector.detect(observation, estimate_tag_pose=False)
    print(f'Number of tags found: {len(tags)}')
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print(f'Insufficient valid tag corners found.')
        return None

    # Get Transformation
    success, rotation_vec, translation = cv2.solvePnP(world_points, image_points, camera_intrinsic, None)
    if success is not True:
        print('PnP Calculation Failed.')
        return None
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    transform_mat = numpy.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation.flatten()

    return transform_mat
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
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        cv2.namedWindow('Verifying World Origin', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying World Origin', 1280, 720)
        cv2.imshow('Verifying World Origin', cv_image)
        cv2.waitKey(0)
    
    finally:
        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()

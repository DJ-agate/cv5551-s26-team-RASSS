import cv2, numpy, time, torch
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot


from matplotlib.colors import to_rgb

CUBE_SIZE = 0.025
GRIPPER_LENGTH = 0.067 * 1000
TCP_OFFSET = [8,-1,GRIPPER_LENGTH,0,0,0]



def get_cube_segmentation(observation, target="red block"):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function leverages text prompts to semantically segment a specific 
    cube (e.g., 'red cube') and determines the cube's pose using its 3D point cloud.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud], where image is the 
        RGB/BGRA array and point_cloud is the registered 3D point cloud.
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
        If no matching object is segmented, returns None.
    """
    
    #instantiate Segmentation model
    model = build_sam3_image_model(checkpoint_path="/home/rob/sam3/checkpoints/sam3.pt")
    processor = Sam3Processor(model)
    

    image, point_cloud = observation
    inf_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    inf_img = Image.fromarray(inf_img)
    print("inf img generated")
    inference_state = processor.set_image(inf_img)
    output = processor.set_text_prompt(state=inference_state, prompt=target)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]



    # img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lb = numpy.array([85, 130, 0])
    # upb = numpy.array([130, 255, 255])

    # lr = numpy.array([0, 120, 70])
    # upr = numpy.array([10, 255, 255])

    # lg = numpy.array([74, 60, 60])
    # upg = numpy.array([84,250,250])
    

    cube_pcds = []
    for mask in masks:
        # print(image.shape)
        # print(mask.detach().cpu().numpy().shape)
        # print(inf_img.shape)
        reshaped_mask = mask.detach().cpu().numpy().reshape(1242,2208)        
        
        masked_points = point_cloud[reshaped_mask]
        # print(masked_points)


        masked_points = masked_points.reshape((-1,4))[0:, 0:3]
        masked_points = masked_points[numpy.all(~numpy.isnan(masked_points), axis=1)]
        
        masked_pcd = o3d.geometry.PointCloud()
        masked_pcd.points = o3d.utility.Vector3dVector(masked_points) #slice to only include xyz vals

        masked_pcd = masked_pcd.voxel_down_sample(voxel_size=5)
        # masked_pcd = masked_pcd.
        masked_pcd, _ = masked_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        cube_pcds.append(masked_pcd)


    
    return cube_pcds

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic



    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud
        
        #instantiate Segmentation model
        # model = build_sam3_image_model(checkpoint_path="/home/rob/sam3/checkpoints/sam3.pt")
        # processor = Sam3Processor(model)
        # processor.reset_all_prompts()
       
        # cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        # cv2.imshow('Verifying Cube Pose', cv_image)
        # From numpy to Open3D
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        # Create an Open3D visualizer
        print("about to open visual")
        cube_pcds = get_cube_segmentation([cv_image,point_cloud])
        
        o3d.io.write_point_cloud("cube_point_cloud.pcd",cube_pcds[0])
        
        

    
    finally:

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()

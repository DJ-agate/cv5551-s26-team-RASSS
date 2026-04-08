import cv2, numpy, time, torch
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot
import copy

#from segmentation import get_cube_segmentation

TAG_SIZE = 0.08

CUBE_TAG_SIZE = 0.0205

def get_cube_transform(camera_pose,cube_pcd):

    obb = cube_pcd.get_minimal_oriented_bounding_box()
    # obb.set_extent([CUBE_SIZE * 1000, CUBE_SIZE * 1000, CUBE_SIZE * 1000])
    # obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, [CUBE_SIZE * 1000, CUBE_SIZE * 1000, CUBE_SIZE * 1000])
    
    t_cam_cube = numpy.eye(4)

    t_cam_cube[:3, 3] = obb.get_center().flatten() / 1000
    # t_cam_cube[:3, :3] = obbs[0].get_rotation_matrix_from_zyx()
    t_cam_cube[:3, :3] = obb.R #numpy.eye(3) #
    print(obb.extent)
    print(t_cam_cube)
    
    return (camera_pose @ t_cam_cube, t_cam_cube)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic



    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud
        

        
        # From numpy to Open3D
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        # Create an Open3D visualizer
        print("about to open visual")
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)

        cv2.namedWindow('Verifying Robot Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Robot Pose', 1280, 720)
        cv2.imshow('Verifying Robot Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        
        
        # # Get the point cloud data as a numpy array
        point_cloud_data = point_cloud

        # Extract the XYZ data
        xyz_data = point_cloud_data[:, :, :3]
        xyz_data = xyz_data.reshape(-1,3)

        # Create an Open3D point cloud
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(xyz_data)
        o3d.visualization.draw_geometries([point_cloud_o3d],
                                  zoom=0.0,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.0, 2.0, 1.0],
                                  up=[-0.0694, -0.9768, 0.2024])


        # Visualization
        # draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        # cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        # cv2.imshow('Verifying Cube Pose', cv_image)
        # key = cv2.waitKey(0)
        pcd = o3d.io.read_point_cloud("cube_point_cloud.pcd")
        # pcd.paint_uniform_color([0,1,0])
        # pcd.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # # Add the point cloud to the visualizer
        # o3d.visualization.draw_geometries([pcd],
        #                           zoom=0.3412,
        #                           front=[0.4257, -0.2125, -0.8795],
        #                           lookat=[2.0, 2.0, 1.0],
        #                           up=[-0.0694, -0.9768, 0.2024])
        
        t_cam_cube = None
        
        
        obb = pcd.get_minimal_oriented_bounding_box()
        

        # o3d.visualization.draw_geometries([pcd, obb],
        #                           zoom=0.33,
        #                           front=[0.5439, -0.2333, -0.8060],
        #                           lookat=[2.0, 2.0, 1.0],
        #                           up=[-0.07, -0.97, 0.2])
        t_cam_cube = numpy.eye(4)

        t_cam_cube[:3, 3] = obb.get_center().flatten() / 1000
        # t_cam_cube[:3, :3] = obbs[0].get_rotation_matrix_from_zyx()
        t_cam_cube[:3, :3] = obb.R #numpy.eye(3) #
        print(obb.extent)
        print(t_cam_cube)
        
        t_robot_cube = numpy.linalg.inv(t_cam_robot) @ t_cam_cube
        print(t_cam_cube)
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()
        

        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.scale(25, center=mesh.get_center())
        mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh])
        v_cube_pcd = mesh.sample_points_uniformly(number_of_points=500)
        o3d.visualization.draw_geometries([v_cube_pcd])

        source = v_cube_pcd
        target = pcd
        threshold = 0.02
        trans_init = numpy.asarray([[1.0, 0.0,  0.0, 0.0],
                                    [0.0, 1.0,    0.0, 0.0],
                                    [0.0, 0.0,  1.0, 0.0],
                                    [0.0, 0.0,  0.0, 1.0]])
        draw_registration_result(source, target, trans_init)


        
    
    finally:

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()

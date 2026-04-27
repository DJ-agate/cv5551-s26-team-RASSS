import cv2, numpy, time, torch
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
#from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from transforms import get_transform_camera_robot
import copy

#from segmentation import get_cube_segmentation

TAG_SIZE = 0.08

CUBE_TAG_SIZE = 0.0205

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source, target):

    source = source
    target = target
    trans_init = numpy.eye(4)
    source.transform(trans_init)
    draw_registration_result(source, target, numpy.eye(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999))
    return result



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
        cv_image_copy = copy.deepcopy(cv_image)
        
      
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        cv2.namedWindow('Verifying Robot Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Robot Pose', 1280, 720)
        cv2.imshow('Verifying Robot Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()
        
        
        # Get the point cloud data as a numpy array
        point_cloud_data = point_cloud

        # Extract the XYZ data
        xyz_data = point_cloud_data[:, :, :3]
        xyz_data = xyz_data.reshape(-1,3)

        # Visualization
        pcd = o3d.io.read_point_cloud("cube_point_cloud.pcd")
        t_cam_cube = None
        obb = pcd.get_minimal_oriented_bounding_box()
        
        t_cam_cube = numpy.eye(4)

        t_cam_cube[:3, 3] = obb.get_center().flatten() / 1000
        t_cam_cube[:3, :3] = obb.R #numpy.eye(3) #
        #print(obb.extent)
        print("t_cam_cube")
        print(t_cam_cube)
        t_robot_cube = numpy.linalg.inv(t_cam_robot) @ t_cam_cube
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
             cv2.destroyAllWindows()

        
        # CREATING V CUBE
        #mesh = o3d.geometry.TriangleMesh.create_box()
        #mesh.scale(30, center=mesh.get_center())

        mesh = o3d.io.read_triangle_mesh("Mug_w_tags.stl")
        mesh.compute_vertex_normals()
        mesh.scale(scale=22,center=mesh.get_center().flatten()/1000)
        # o3d.visualization.draw_geometries([mesh])
        v_cube_pcd = mesh.sample_points_uniformly(number_of_points=45000)
        #v_cube_pcd = v_cube_pcd.translate([0,0,0],relative=False)

        # o3d.visualization.draw_geometries([v_cube_pcd])
        print("init_v_cube_default_transform")
        print(v_cube_pcd.get_center())
        


        init_cam_cube = numpy.eye(4)
        init_cam_cube[:3, 3] = pcd.get_center().flatten()/1000
        #init_cam_cube[:3, :3] = obb.R 
        print("to_init_v_cube_transform")
        print(init_cam_cube)



        target = pcd.translate(pcd.get_center().flatten()/1000, relative=False)


        voxel_size = 5
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            voxel_size,v_cube_pcd,target)
        
        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)

        print("CENTER OF DEPTH PCD")
        print(target.get_center())
        threshold = 50
        trans_init = init_cam_cube

        # ICP REGISTRATION (REFINEMENT)
        reg_p2p = o3d.pipelines.registration.registration_icp(
                source,target, threshold, result_ransac.transformation,
                #o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        print(reg_p2p)
        print(reg_p2p.transformation)
        draw_registration_result(source, target, reg_p2p.transformation)
        icp_pose_est = numpy.eye(4)
        icp_pose_est[:3, 3] = reg_p2p.transformation[:3, 3]/1000
        icp_pose_est[:3, :3] = reg_p2p.transformation[:3, :3]
        
        print("icp_pose_est")
        print(icp_pose_est)
        draw_pose_axes(cv_image_copy, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        draw_pose_axes(cv_image_copy, camera_intrinsic, trans_init @ icp_pose_est) # I am unsure if this is the correct way the pose is done, however, it works?
        cv2.namedWindow('new Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('new Cube Pose', 1280, 720)
        cv2.imshow('new Cube Pose', cv_image_copy)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()
    
    finally:
        print("done")
        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()

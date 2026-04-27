import pickle

from utils.zed_camera import ZedCamera
import cv2, time


from utils.vis_utils import draw_grasp_poses
from utils.vis_utils import plot_trajectory
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import RigidTransform, Rotation
import numpy as np
from xarm.wrapper import XArmAPI
from objective import objective_optimizer
import trimesh
from sdf_visualization import visualize_workspace

#for robot frame
TAG_SIZE = 0.08
GRIPPER_LENGTH = 0.069 * 1000
TCP_OFFSET = [0,0,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205

robot_ip = '192.168.1.172'
INIT_ARM = [86.954865, 0.818719, 85.287003, 179.991483, -0.001547, 0.001776]
INIT_POSE = RigidTransform.from_components(INIT_ARM[:3], Rotation.from_euler('XYZ', INIT_ARM[3:], degrees=True))

def main():
    mug_poses = np.load("poses.pkl", allow_pickle=True)
    print("mug poses pkl:")
    print(mug_poses[0])
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    time.sleep(2.5)

    try:
        # Get Observation
        cv_image = zed.image

        
        cv_image = zed.image

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        # if t_cam_robot is None:
        #     return
        t_cam_tag = None
        t_robot_cube , t_cam_tag, tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot))
        #print(t_cam_tag)
        #import get_mug_transform

        t_cam_mug = get_mug_from_april(t_cam_tag, tag_id) #todo once complete
        
        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug

        t_robot_mug[:3, 3] = t_robot_mug[:3, 3] *1000
        
        # Visualize the mug with the SDF
        # worspace_boundary = [[0, 0.380], [-0.400, 0.400], [0, 0.500]]
        # visualize_workspace(np.copy(t_robot_mug), workspace_bound=worspace_boundary, 
        #                     workspace_resolution=64, display_2d_slices=False, 
        #                     select_specific_dist=False, d_star=10, eps=0.2)

        #print("t_robot_mug")
        #print(t_robot_mug)
        '''
        NEEDS TO BE EDIT SAFE! USE .COPY()
        '''
        draw_grasp_poses(cv_image, camera_intrinsic, np.copy(mug_poses), t_cam_mug )

        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
        if key == ord('k'):
            cv2.destroyAllWindows()

        ##TODO: IMPORTANT to transform all mug poses before passing them in
        
        print("t_mug_grasp")
        print(mug_poses[0])
        grasp_pose = t_robot_mug @ mug_poses[0]
        print("t_robot_mug_grasp = ", grasp_pose)

        mesh = trimesh.load_mesh("Mug_wo_tags.stl")
        mesh.apply_scale(1.0)
        
        T_mug_robot = RigidTransform.from_matrix(np.linalg.inv(t_robot_mug))

        p_mug_q = T_mug_robot * INIT_POSE
        print("p_mug_q")
        print(p_mug_q)
        

        p_mug_q = [p_mug_q.translation.T]

        value = trimesh.proximity.signed_distance(mesh, p_mug_q)[0]
        closest_point = trimesh.proximity.closest_point(mesh, p_mug_q)[0]

        print("value")
        print(value)
        print("closest_point")
        print(closest_point)

        
    finally:
        # Close Lite6 Robot
        #arm.move_gohome(wait=True)
        time.sleep(0.5)
        # arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
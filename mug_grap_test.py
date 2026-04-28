import pickle

from utils.zed_camera import ZedCamera
import cv2, time


from utils.vis_utils import draw_grasp_poses
from utils.vis_utils import plot_trajectory
from transforms import get_transform_camera_robot, get_transform_cube, get_mug_from_april
from scipy.spatial.transform import Rotation as R
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
INIT_POSE = [86.954865, 0.818719, 85.287003, 179.991483, -0.001547, 0.001776]



def main():
    mug_poses = np.load("poses.pkl", allow_pickle=True)
    print("mug poses pkl:")
    print(mug_poses[0])
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    objects = []
    #mug to grab mesh
    mesh = trimesh.load_mesh("Mug_wo_tags.stl")
    mesh.apply_scale(1000.0)
    objects.append(mesh)
    #create and append tower
    objects.append(trimesh.creation.box(extents=[20.5,20.5,4*20.5]))
    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset(TCP_OFFSET)
    arm.set_mode(0)
    arm.set_state(0)
    # arm.move_gohome(wait=True)
    time.sleep(2.5)

    try:
        # Get Observation
        cv_image = zed.image

        # Get Mug transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        t_cam_tag = None
        t_robot_cube , t_cam_tag, mug_tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot), [5,9])
        t_cam_mug = get_mug_from_april(t_cam_tag, mug_tag_id) #todo once complete
        t_robot_mug = np.linalg.inv(t_cam_robot) @ t_cam_mug
        t_robot_mug[:3, 3] = t_robot_mug[:3, 3] *1000
        
        t_cam_tag = None
        t_robot_cube , t_cam_tag, tower_tag_id = get_transform_cube(cv_image, camera_intrinsic, np.linalg.inv(t_cam_robot), [4,4])

        t_robot_tower = np.linalg.inv(t_cam_robot) @ t_cam_tag
        t_robot_tower[:3, 3] = t_robot_tower[:3, 3] *1000
        t_robot_tower[2][3] = t_robot_tower[2][3] - (20.5*2)

        # Visualize the mug with the SDF
        worspace_boundary = [[0, 0.380], [-0.400, 0.400], [0, 0.500]]
        visualize_workspace(np.copy(t_robot_mug), workspace_bound=worspace_boundary, 
                            workspace_resolution=64, display_2d_slices=False, 
                            select_specific_dist=False, d_star=10, eps=0.2)

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
        
        # trimesh.creation.cylinder

        T_mug_robot = np.linalg.inv(t_robot_mug)
        t_tower_robot = np.linalg.inv(t_robot_tower)
        T_objects_robot = [np.copy(T_mug_robot),np.copy(t_tower_robot)]
        
        obj_opt = objective_optimizer(arm.get_position()[1],[grasp_pose],objects,T_objects_robot)
        # obj_opt = objective_optimizer(INIT_POSE,[grasp_pose],[mesh],T_mug_robot)
        trajectory= obj_opt.get_euler_trajectory()
        print("arm initial:")
        print(arm.get_position()[1])
        print("trajectory: ")
        print(trajectory)

        
        plot_trajectory(np.copy(trajectory))
        obj_opt.optimize_trajectory()
        trajectory= obj_opt.get_euler_trajectory()
        plot_trajectory(np.copy(trajectory))
        arm.move_arc_lines(trajectory)

        
        worspace_boundary = [[0, 0.380], [-0.400, 0.400], [0, 0.500]]
        visualize_workspace(np.copy(t_robot_mug), workspace_bound=worspace_boundary, 
                            workspace_resolution=32, display_2d_slices=False, 
                            select_specific_dist=False, d_star=10, eps=0.2, trajectory=obj_opt.trajectory.copy())
        #print(obj_opt.trajectory)


        # Grasp
        # x = grasp_pose[0][3]
        # y = grasp_pose[1][3]
        # z = grasp_pose[2][3]
        
        # rot_pose = np.eye(3)
        # rot_pose[:3][:3] = grasp_pose[0:3,0:3]
        # rot = R.from_matrix(rot_pose)
        # angles = rot.as_euler("xyz",degrees=True)
        # roll = angles[0]
        # pitch = angles[1]
        # yaw = angles[2]
        # arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True)

        # for i in range(len(obj_opt.trajectory)):
        #     grasp_pose = obj_opt.trajectory[i]
        #     x = grasp_pose[0]
        #     y = grasp_pose[1]
        #     z = grasp_pose[2]
            
        #     # rot_pose = np.eye(3)
        #     # rot_pose[:3][:3] = grasp_pose[0:3,0:3]
        #     # rot = R.from_matrix(rot_pose)
            
        #     roll = grasp_pose[3]
        #     pitch = grasp_pose[4]
        #     yaw = grasp_pose[5]
                
            
        #     arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True)
        #     time.sleep(1)

        #arm.open_lite6_gripper()
        time.sleep(0.5)
        # arm.set_position(x,y,z,roll,pitch,yaw,is_radian=None,wait=True, speed=200)


        
    
    finally:
        # Close Lite6 Robot
        #arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
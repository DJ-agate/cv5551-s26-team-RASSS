import cv2, numpy, pickle
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def draw_pose_axes(image, camera_intrinsic, pose, size=0.1):

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    # origin and 3 unit vector of the frame
    frame_points = numpy.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]).reshape(-1,3) * size

    ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    ipoints = numpy.round(ipoints).astype(int)

    origin = tuple(ipoints[0].ravel())
    unit_x = tuple(ipoints[1].ravel())
    unit_y = tuple(ipoints[2].ravel())
    unit_z = tuple(ipoints[3].ravel())

    cv2.line(image, origin, unit_x, (0,0,255), 2)
    cv2.line(image, origin, unit_y, (0,255,0), 2)
    cv2.line(image, origin, unit_z, (255,0,0), 2)

def draw_grasp_poses(image, camera_intrinsic, grasp_poses, mug_pose, size=0.1):
    #draw mug frame
    draw_pose_axes(image, camera_intrinsic, mug_pose, size=0.1)

    #draw grasp frames
    for grasp_pose in grasp_poses:
        # grasp pose in camera frame
        grasp_pose_cam = mug_pose @ grasp_pose
        rvec, _ = cv2.Rodrigues(grasp_pose[:3,:3])
        tvec = grasp_pose[:3, 3]

        # origin and points for gripper
        frame_points = numpy.array([[0, 0, 0],
                                    [0, 0, 0.5],
                                    [0.25, 0, 0],
                                    [-0.25, 0, 0]
                                    [0.25, 0, -0.5],
                                    [-0.25, 0. -0.25]]).reshape(-1,3) * size

        ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
        ipoints = numpy.round(ipoints).astype(int)

        origin = tuple(ipoints[0].ravel())
        top_point = tuple(ipoints[1].ravel())
        side_1 = tuple(ipoints[2].ravel())
        side_2 = tuple(ipoints[3].ravel())
        side_1_fork = tuple(ipoints[4].ravel())
        side_2_fork = tuple(ipoints[5].ravel())
        cv2.line(image, origin, top_point, (0,255,255), 2)
        cv2.line(image, origin, side_1, (0,255,255), 2)
        cv2.line(image, origin, side_2, (0,255,255), 2)
        cv2.line(image, side_1, side_1_fork, (0,255,255), 2)
        cv2.line(image, side_2, side_2_fork, (0,255,255), 2)


def plot_trajectory(trajectory):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(trajectory[0], trajectory[1], trajectory[2])

    ax.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

    plt.show()

def draw_trajectory(image, camera_intrinsic, pose, trajectory, size=0.1):
    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    # origin and 3 unit vector of the frame
    frame_points = numpy.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]).reshape(-1,3) * size

    ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    ipoints = numpy.round(ipoints).astype(int)

    origin = tuple(ipoints[0].ravel())
    unit_x = tuple(ipoints[1].ravel())
    unit_y = tuple(ipoints[2].ravel())
    unit_z = tuple(ipoints[3].ravel())
    
    cv2.line(image, origin, unit_x, (0,0,255), 2)
    cv2.line(image, origin, unit_y, (0,255,0), 2)
    cv2.line(image, origin, unit_z, (255,0,0), 2)
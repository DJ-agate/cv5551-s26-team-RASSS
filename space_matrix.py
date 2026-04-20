import numpy as np
import cv2
from pupil_apriltags import Detector
import open3d as o3d


"""
Create the 3D space matrix of points representing the entire workspace of the
robot and the mug. This will be used to create the distance function points
that the robot will follow to reach the mug grapsing points
"""

"""
Generate space matrix
Input:
    x_range: tuple (x_min, x_max)
    y_range: tuple (y_min, y_max)
    z_range: tuple (z_min, z_max)
    step: float, the distance between points in the space matrix
Output:
    space_matrix: numpy array of shape (n, 3) representing the 3D space matrix
"""
def gen_space_matrix(x_range, y_range, z_range, step):
    x_points = np.arange(x_range[0], x_range[1], step)
    y_points = np.arange(y_range[0], y_range[1], step)
    z_points = np.arange(z_range[0], z_range[1], step)

    space_matrix = np.array(np.meshgrid(x_points, y_points, z_points)).T.reshape(-1, 3)
    return space_matrix

"""
Visualize the space matrix using Open3D
Input:
    space_matrix: numpy array of shape (n, 3) representing the 3D space matrix  
"""
def visualize_space_matrix(space_matrix):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(space_matrix)
    o3d.visualization.draw_geometries([pcd])

"""
Visualize a slice of the space matrix at a given z value
Input:  
    space_matrix: numpy array of shape (n, 3) representing the 3D space matrix
    z_value: float, the z value at which to visualize the slice
    threshold: float, the distance threshold to include points in the slice
"""
def visualize_space_matrix_slice(space_matrix, z_value, threshold):
    slice_points = space_matrix[np.abs(space_matrix[:, 2] - z_value) < threshold]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(slice_points)
    o3d.visualization.draw_geometries([pcd])

"""
Set the floor of the space matrix to a border value (like 0 for unsigned distance function
or a large positive value for signed distance function) to ensure the robot does not go below the floor
Input:
    space_matrix: numpy array of shape (n, 3) representing the 3D space matrix
    floor_z: float, the z value representing the floor level
    border_value: float, the value to set for points below the floor    
Output:
    modified_space_matrix: numpy array of shape (n, 3) representing the modified space matrix
"""
def set_floor_border(space_matrix, floor_z, border_value):
    modified_space_matrix = space_matrix.copy()
    modified_space_matrix[modified_space_matrix[:, 2] < floor_z, 2] = border_value
    return modified_space_matrix

"""
Set mug pose points in the space matrix to a border value (like 0 for unsigned distance function)
to ensure robot is attracted to the mug grasping points
Input:
    space_matrix: numpy array of shape (n, 3) representing the 3D space matrix
    mug_pose: numpy array of shape (4, 4) representing the transformation matrix of the mug pose
    grasping_points: list of numpy arrays of shape (3,) representing the grasping points in the mug frame
    border_value: float, the value to set for points at the mug grasping points
"""
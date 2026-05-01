import open3d as o3d
import numpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

### Constants ###
workspace_bound = [[0, 0.38], [-0.4, 0.4], [0, 0.5]]

"""
SDF and UDF Visualization Querys
"""
# Query SDF/sDF
def query_sdf(scene, point):
    pts = o3d.core.Tensor(point,dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(pts)
    return sdf.numpy()

def query_udf(scene, points):
    pts = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(pts)
    return d.numpy()

"""
Compute workspace boundary based on the center point of the mug in the world frame and the predefined workspace boundary matrix
Parameters:
    mug_transform: the 4x4 transformation matrix of the mug in the world frame
    workspace_bound: a matrix containing the workspace boundary matrix as 
    [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
Returns:
    workspace_bound: a matrix containing the workspace boundary matrix updated for
    the center point of the mug in the world frame
"""
def compute_workspace_bound(mug_center, workspace_bound):
    # Get the center point of the mug
    # Update the workspace boundary based on the center point of the mug
    # X_min
    workspace_bound[0][0] = workspace_bound[0][0] - mug_center[0]
    # X_max
    workspace_bound[0][1] = workspace_bound[0][1] - mug_center[0]
    # Y_min 
    workspace_bound[1][0] = workspace_bound[1][0] - mug_center[1]
    # Y_max
    workspace_bound[1][1] = workspace_bound[1][1] - mug_center[1]
    # Z_min
    workspace_bound[2][0] = workspace_bound[2][0] - mug_center[2]
    # Z_max
    workspace_bound[2][1] = workspace_bound[2][1] - mug_center[2]
    return workspace_bound

"""
Compute the 3D grid points for visualization
"""
def compute_3d_points(workspace_bound, workspace_resolution):
    # Get the boundary values
    x_min, x_max = workspace_bound[0]
    y_min, y_max = workspace_bound[1]
    z_min, z_max = workspace_bound[2]

    x = numpy.linspace(x_min, x_max, workspace_resolution)
    y = numpy.linspace(y_min, y_max, workspace_resolution)
    z = numpy.linspace(z_min, z_max, workspace_resolution)

    # 3D grid
    grid = numpy.stack(numpy.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(numpy.float32)

    # flatten -> query -> reshape
    points = grid.reshape(-1, 3)
    return points

"""
Get the 2D slices of the SDF/UDF values in the workspace for visualization
Parameters:
    sdf: the 3D array containing the SDF/UDF values in the workspace
    xyz_s;lices: the x, y, and z indices for 2d slices as a list
    display_2d_slices: whether to display 2D slices of the SDF/UDF values 
Returns:
    (none)
"""
def visualize_2d_slices(sdf, xyz_slices, display_2d_slices):
    plt.imshow(sdf[:, :, int(xyz_slices[2])], cmap='jet')
    plt.colorbar()
    plt.title("SDF slice (z)")
    plt.savefig("sdf_slice_z.png")
    if display_2d_slices:
        plt.show()

    plt.imshow(sdf[int(xyz_slices[0]), :, :], cmap='jet')
    plt.colorbar()
    plt.title("SDF slice (x)")
    plt.savefig("sdf_slice_x.png")
    if display_2d_slices:
        plt.show()

    plt.imshow(sdf[:, int(xyz_slices[1]), :], cmap='jet')
    plt.colorbar()
    plt.title("SDF slice (y)")
    plt.savefig("sdf_slice_y.png")
    if display_2d_slices:
        plt.show()

"""
Visualize the Mug in the 3D space

Parameters: 
    mug_transform: the 4x4 transformation matrix of the mug in the world frame
    workspace_bound: a matrix containing the workspace boundary matrix as 
    [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    workspace_resolution: the resolution of the workspace grid for visualization
    display_2d_slices: whether to display 2D slices of the SDF/UDF values in the workspace
Returns:
    (none)
"""
def visualize_workspace(mug_transform, workspace_bound=None, workspace_resolution=64, display_2d_slices=True, select_specific_dist=False, d_star=0.01, eps=0.002, trajectory=None, obstacle=False, obst_transform=None):
    rotation_matrix = mug_transform[:3, :3]
    rot_90 = Rotation.from_euler('XYZ',[0,0,-90], degrees=True).as_matrix()
    # rotation_matrix = rotation_matrix@rot_90
    rotation_xyz = Rotation.from_matrix(rotation_matrix).as_euler('XYZ', degrees=True)
    translation_vector = mug_transform[:3, 3]
    if translation_vector[0] > 0.5:
        # change into meters
        for i in range(3):
            translation_vector[i] = translation_vector[i] / 1000
    
    mesh_legacy = o3d.io.read_triangle_mesh("SOLID_mug_wo_tags.stl")
    # Update to new format
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    mesh.compute_vertex_normals()
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    mesh_legacy = mesh_legacy.rotate(rotation_matrix)
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    
    # update workspace_bound based on the center point of the mug
    workspace_bound = compute_workspace_bound(translation_vector, workspace_bound)

    points = compute_3d_points(workspace_bound, workspace_resolution)
    
    geom_list = []
    # add the obstacle if there is one
    if obstacle==True:
        print("Adding obstacle to visual)")
        obst_transform[:3,3] = obst_transform[:3,3]/1000
        obst_transform = numpy.linalg.inv(mug_transform)@obst_transform
        obst_t = obst_transform[:3,3]
        obst_rot = obst_transform[:3,:3]
        obst_rot = obst_rot@rot_90
        # if obst_t[0] > 0.5:q
        #     # change into meters
        #     for i in range(3):
        #         obst_t[i] = obst_t[i] / 1000
        obst_xyz = Rotation.from_matrix(obst_rot).as_euler('XYZ', degrees=True)

        obstacle_mesh = o3d.io.read_triangle_mesh("Obstacle.stl")
        obstacle_legacy = o3d.t.geometry.TriangleMesh.from_legacy(obstacle_mesh)
        obstacle_mesh.compute_vertex_normals()
        # obstacle_legacy.scale(100.0, [0,0,0])
        # obstacle_mesh.rotate(obst_rot)

        obstacle_mesh.translate(obst_t)
        _ = scene.add_triangles(obstacle_legacy)
        geom_list.append(obstacle_mesh)

    sdf = query_sdf(scene, points)  ### SDF/sDF
    sdf = sdf.reshape(workspace_resolution, workspace_resolution, workspace_resolution)

    slice_2d_resolution = workspace_resolution*2

    # Visualize - 2D slice
    points_2d = compute_3d_points(workspace_bound, workspace_resolution*2)  # for 2D slice visualization
    sdf_2d = query_sdf(scene, points_2d)  ### SDF/sDF
    sdf_2d = sdf_2d.reshape(slice_2d_resolution, slice_2d_resolution, slice_2d_resolution)
    xyz_slices = [slice_2d_resolution// 2, slice_2d_resolution // 2, slice_2d_resolution // 2]
    visualize_2d_slices(sdf_2d, xyz_slices, display_2d_slices)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=(0,0.4,0))


    if trajectory is not None:
        geom_list = add_trajectory(mug_transform.copy(), trajectory)
    geom_list.append(mesh_frame)
    # # Visualize the mug and the workspace
    # o3d.visualization.draw_geometries([mug_frame, workspace_box])
    if select_specific_dist == True:
        mask = (numpy.abs(sdf - d_star) < eps).reshape(-1)
        contact_points = points[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(contact_points)
        pcd.paint_uniform_color([1, 0, 0])

    else:
        ## Visualize - 3D point cloud
        sdf_norm = (sdf - sdf.min()) / (sdf.max() - sdf.min())
        colors = plt.cm.jet(sdf_norm.flatten())[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # make it more visually appealing
        mesh_legacy.compute_vertex_normals()
        # Draw the 3d Mesh
    geom_list.append(pcd)
    geom_list.append(mesh_legacy)

    # add the obstacle if there is one
    if obstacle==True:
        print("Adding obstacle to visual)")
        obst_t = obst_transform[:3,3]
        obst_rot = obst_transform[:3,:3]
        if obst_t[0] > 0.5:
            # change into meters
            for i in range(3):
                obst_t[i] = obst_t[i] / 1000
        obst_xyz = Rotation.from_matrix(obst_rot).as_euler('XYZ', degrees=True)

        obstacle_mesh = o3d.io.read_triangle_mesh("Obstacle.stl")
        obstacle_legacy = o3d.t.geometry.TriangleMesh.from_legacy(obstacle_mesh)
        obstacle_mesh.compute_vertex_normals()
        # obstacle_legacy.scale(1000.0, [0,0,0])
        obstacle_mesh.rotate(obst_rot)
        obstacle_mesh.translate(obst_t)
        _ = scene.add_triangles(obstacle_legacy)
        geom_list.append(obstacle_mesh)

    o3d.visualization.draw_geometries(geom_list)

def add_trajectory(mug_transform, trajectory):
    pose_geoms = [o3d.geometry.TriangleMesh.create_box(height=0.01, width=0.01, depth=0.01) for _ in range(len(trajectory))]
    
    rotation_matrix = mug_transform[:3, :3]
    rot_90 = Rotation.from_euler('XYZ',[0,45,0], degrees=True).as_matrix()
    # rotation_matrix = rotation_matrix@rot_90
    translation_vector = mug_transform[:3, 3]
    if translation_vector[0] > 0.5:
        # change into meters
        for i in range(3):
            translation_vector[i] = translation_vector[i] / 1000
    
    # mesh_legacy = mesh_legacy.rotate(rotation_matrix)
    
    
    for i in range(len(pose_geoms)):
        pose_mat = trajectory[i].as_matrix()
        pose_mat[:3, 3] /= 1000
        pose_mat = numpy.linalg.inv(pose_mat) @ mug_transform #(mug_transform) @ pose_mat
        # pose_mat[0,3] *= -1
        # x=pose_mat[0][3]
        # pose_mat[0][3] = pose_mat[1][3]
        # pose_mat[1][3] = x
        pose_geoms[i].paint_uniform_color([0,0,0])
        pose_geoms[i] = pose_geoms[i].transform(pose_mat)
        pose_geoms[i] = pose_geoms[i].rotate(rot_90)
        
    return pose_geoms
    

# #############MAIN#################
# mesh_legacy = o3d.io.read_triangle_mesh("Mug_wo_tags.stl")
# # Update to new format
# mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
# mesh.compute_vertex_normals()
# # Create a scene and add the triangle mesh
# scene = o3d.t.geometry.RaycastingScene()
# _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh



# center = mesh_legacy.get_center()
# print("center:", center)
# center = numpy.array([0.2, 0.3, 0.039])  # for visualization purposes, we can set the center to be the middle of the workspace
# _ = visualize_workspace(center, workspace_bound, workspace_resolution=64, display_2d_slices=False, select_specific_dist=False, d_star=0.2);
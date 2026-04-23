import open3d as o3d
import numpy
import matplotlib.pyplot as plt

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
Visualize the Mug in the 3D space

Parameters: 
    mug_transform: the 4x4 transformation matrix of the mug in the world frame
    workspace_bound: a matrix containing the workspace boundary matrix as 
    [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    workspace_resolution: the resolution of the workspace grid for visualization
Returns:
    (none)
"""
def visualize_workspace(mug_transform, workspace_bound=None, workspace_resolution=64):
    
    # Create a coordinate frame for the mug
    # mug_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # mug_frame.transform(mug_transform)

    # # Create a bounding box for the workspace
    # x_min, x_max = workspace_bound[0]
    # y_min, y_max = workspace_bound[1]
    # z_min, z_max = workspace_bound[2]

    # workspace_box = o3d.geometry.AxisAlignedBoundingBox(
    #     min_bound=[x_min, y_min, z_min],
    #     max_bound=[x_max, y_max, z_max]
    # )
    # workspace_box.color = (1, 0, 0)  # Red color
    
    min_bound = mesh_legacy.get_min_bound()
    max_bound = mesh_legacy.get_max_bound()

    # update workspace_bound based on the center point of the mug
    workspace_bound = compute_workspace_bound(mug_transform, workspace_bound)

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

    sdf = query_sdf(scene, points)  ### SDF/sDF
    sdf = sdf.reshape(workspace_resolution, workspace_resolution, workspace_resolution)

    # Visualize - 2D slice
    mid_z = workspace_resolution // 2
    mid_x = workspace_resolution // 2
    mid_y = workspace_resolution // 2

    plt.imshow(sdf[:, :, mid_z], cmap='jet')
    plt.colorbar()
    plt.title("sDF slice (z mid)")
    plt.savefig("sdf_slice_z.png")
    plt.show()

    plt.imshow(sdf[mid_x, :, :], cmap='jet')
    plt.colorbar()
    plt.title("sDF slice (x mid)")
    plt.savefig("sdf_slice_x.png")
    plt.show()

    plt.imshow(sdf[:, mid_y, :], cmap='jet')
    plt.colorbar()
    plt.title("sDF slice (y mid)")
    plt.savefig("sdf_slice_y.png")
    plt.show()

    # # Visualize the mug and the workspace
    # o3d.visualization.draw_geometries([mug_frame, workspace_box])
    ## Visualize - 3D point cloud
    sdf_norm = (sdf - sdf.min()) / (sdf.max() - sdf.min())
    colors = plt.cm.jet(sdf_norm.flatten())[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # make it more visually appealing
    mesh_legacy.compute_vertex_normals()
    # Draw the 3d Mesh
    o3d.visualization.draw_geometries([pcd, mesh_legacy])

#############MAIN#################
mesh_legacy = o3d.io.read_triangle_mesh("Mug_wo_tags.stl")
# Update to new format
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
mesh.compute_vertex_normals()
# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh



center = mesh_legacy.get_center()
print("center:", center)
center = numpy.array([0.3, -0.3, 0.039])  # for visualization purposes, we can set the center to be the middle of the workspace
_ = visualize_workspace(center, workspace_bound, workspace_resolution=64);
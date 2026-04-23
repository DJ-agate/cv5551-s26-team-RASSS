import open3d as o3d
import numpy
import matplotlib.pyplot as plt

# Load mesh and convert to open3d.t.geometry.TriangleMesh
mesh_legacy = o3d.io.read_triangle_mesh("Mug_w_tags.stl")
# mesh_legacy.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

center = mesh_legacy.get_center()
# print("center:", center)

# Query SDF/UDF
def query_sdf(scene, point):
    pts = o3d.core.Tensor(point,dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(pts)
    return sdf.numpy()

def query_udf(scene, points):
    pts = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(pts)
    return d.numpy()

points = numpy.array([
    [0,0,0],       # inside
    [0.1,0,0],     # outside
    [-0.12216329, -0.01450637, -0.44469014]    # center
], dtype=numpy.float32)

# print(query_udf(scene, points))

# print("is_empty:", mesh_legacy.is_empty())
# print("is_watertight:", mesh_legacy.is_watertight())
# print("is_edge_manifold:", mesh_legacy.is_edge_manifold())
# print("is_vertex_manifold:", mesh_legacy.is_vertex_manifold())

# print("min bound:", mesh_legacy.get_min_bound())
# print("max bound:", mesh_legacy.get_max_bound())
# print("center:", mesh_legacy.get_center())


min_bound = mesh_legacy.get_min_bound()
max_bound = mesh_legacy.get_max_bound()

# resolution
N = 64

x = numpy.linspace(min_bound[0], max_bound[0], N)
y = numpy.linspace(min_bound[1], max_bound[1], N)
z = numpy.linspace(min_bound[2], max_bound[2], N)

# 3D grid
grid = numpy.stack(numpy.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(numpy.float32)

# flatten -> query -> reshape
points = grid.reshape(-1, 3)

udf = query_udf(scene, points)
udf = udf.reshape(N, N, N)

# Visualize - 2D slice
mid_z = N // 2

plt.imshow(udf[:, :, mid_z], cmap='jet')
plt.colorbar()
plt.title("UDF slice (z mid)")
plt.show()


## Select all d=0.01 area
# d_star = 0.01
# eps = 0.002
# mask = (numpy.abs(udf - d_star) < eps).reshape(-1)
# contact_points = points[mask]

## "Detect" handle
# min_bound = mesh_legacy.get_min_bound()
# max_bound = mesh_legacy.get_max_bound()
# handle_center = numpy.array([
#     max_bound[0],
#     (min_bound[1] + max_bound[1]) / 2,
#     (min_bound[2] + max_bound[2]) / 2
# ])
# r = 0.3
# mask_handle = numpy.linalg.norm(contact_points - handle_center, axis=1) < r
# contact_points = contact_points[mask_handle]

## Visualize selected area
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(contact_points)
# pcd.paint_uniform_color([1, 0, 0])

# o3d.visualization.draw_geometries([pcd, mesh_legacy])



## Visualize - 3D point cloud
# udf_norm = (udf - udf.min()) / (udf.max() - udf.min())
# colors = plt.cm.jet(udf_norm.flatten())[:, :3]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# mesh_legacy.compute_vertex_normals()

# o3d.visualization.draw_geometries([pcd, mesh_legacy])










# min_bound = mesh_legacy.get_min_bound()
# max_bound = mesh_legacy.get_max_bound()
# bbox_center = (min_bound + max_bound) / 2.0

# test_points = numpy.array([
#     bbox_center,
#     bbox_center + numpy.array([0.0, 0.0, 0.02]),
#     bbox_center + numpy.array([0.0, 0.0, -0.02]),
#     bbox_center + numpy.array([0.02, 0.0, 0.0]),
#     bbox_center + numpy.array([-0.02, 0.0, 0.0]),
# ], dtype=numpy.float32)

# print("signed:", query_sdf(scene, test_points))
# print("unsigned:", query_udf(scene, test_points))

# query_point = o3d.core.Tensor([[10, 10, 10]], dtype=o3d.core.Dtype.Float32)

# # Compute distance of the query point from the surface
# # unsigned_distance = scene.compute_distance(query_point)
# signed_distance = scene.compute_signed_distance(query_point)
# # occupancy = scene.compute_occupancy(query_point)

# # print("unsigned distance", unsigned_distance.numpy())
# print("signed_distance", signed_distance.numpy())
# # print("occupancy", occupancy.numpy())

# min_bound = mesh.vertex['positions'].min(0).numpy()
# max_bound = mesh.vertex['positions'].max(0).numpy()

# N = 256
# query_points = numpy.random.uniform(low=min_bound, high=max_bound,
#                                  size=[N, 3]).astype(numpy.float32)

# # Compute the signed distance for N random points
# signed_distance = scene.compute_signed_distance(query_points)

# xyz_range = numpy.linspace(min_bound, max_bound, num=32)

# # query_points is a [32,32,32,3] array ..
# query_points = numpy.stack(numpy.meshgrid(*xyz_range.T), axis=-1).astype(numpy.float32)

# # signed distance is a [32,32,32] array
# signed_distance = scene.compute_signed_distance(query_points)

# # We can visualize a slice of the distance field directly with matplotlib
# plt.imshow(signed_distance.numpy()[:, :, 15])

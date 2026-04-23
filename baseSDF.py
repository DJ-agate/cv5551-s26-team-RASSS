import numpy 
import trimesh
import matplotlib.pyplot as plt

mesh = trimesh.load(
    "Mug_w_tags.stl", 
    force="mesh",
    process=True,
    validate=True
    )

# Watertight: is closed surface
# Trimesh only works for watertight objects
print("If watertight:", mesh.is_watertight)
print("Raw extents (inch):", mesh.extents)
print("Raw bounds (inch):", mesh.bounds)

# Suppose the unit of STL is inch
# Convert inch into meter
mesh.apply_scale(0.0254)
print("scaled extents (m):", mesh.extents)
print("scaled bounds (m):", mesh.bounds)
print("Bounds corners: ", trimesh.bounds.corners(mesh.bounds))
# print("Edges: ", mesh.edges)
# print("Facets: ", mesh.facets)
print("Centroid: ", mesh.centroid)

# Check the definition of signs (positive/negtive?)
center = mesh.centroid.reshape(1, 3).astype(numpy.float32)
print("Center sdf:", trimesh.proximity.signed_distance(mesh, center))

outside = numpy.array([[0.2, 0.2, 0.2]], dtype=numpy.float32)
print("Outside sdf:", trimesh.proximity.signed_distance(mesh, outside))

# Grid resolution
voxel_size = 0.005   # 5 mm

# Padding outside the bounding box
padding = 0.01       # 1 cm

bounds = mesh.bounds.copy()
bounds[0] -= padding
bounds[1] += padding

xs = numpy.arange(bounds[0, 0], bounds[1, 0] + voxel_size, voxel_size)
ys = numpy.arange(bounds[0, 1], bounds[1, 1] + voxel_size, voxel_size)
zs = numpy.arange(bounds[0, 2], bounds[1, 2] + voxel_size, voxel_size)

nx, ny, nz = len(xs), len(ys), len(zs)
print("grid size:", nx, ny, nz)
print("total points:", nx * ny * nz)

sdf_grid = numpy.empty((nx, ny, nz), dtype=numpy.float32)

chunk = 4   # Handle 4 chunks at a time
for k0 in range(0, nz, chunk):
    k1 = min(k0 + chunk, nz)
    zz = zs[k0:k1]

    X, Y, Z = numpy.meshgrid(xs, ys, zz, indexing="ij")
    query_points = numpy.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    print(f"processing slices {k0}:{k1}, query points = {len(query_points)}")

    sdf_vals = trimesh.proximity.signed_distance(mesh, query_points)
    sdf_grid[:, :, k0:k1] = sdf_vals.reshape(nx, ny, k1 - k0)

# Save npz file
numpy.savez(
    "mug_sdf.numpyz",
    sdf=sdf_grid,
    xs=xs,
    ys=ys,
    zs=zs,
    voxel_size=voxel_size
)

# print("Saved mug_sdf.numpyz")
print("sdf min/max:", sdf_grid.min(), sdf_grid.max())


# Visualization
data = numpy.load("mug_sdf.npz")
sdf = data["sdf"]
# sdf.shape is a 3D array (x, y, z)
# Use the z-axis sdf to visualize, slice from the middle
mid_z = sdf.shape[2] // 2
plt.figure()
plt.imshow(sdf[:, :, mid_z].T, origin="lower")
plt.colorbar()
plt.title("SDF mid-z slice")
plt.savefig("baseSDF.png", dpi=300, bbox_inches='tight')
plt.show()


# Get transformation (model frame -> cam frame)
handle = mesh.centroid 
handle[0] = handle[0] + 0.06  
roll = mesh.centroid - handle
yaw = numpy.array([0, 0, 1]) # just suppose 
pitch = numpy.cross(roll, yaw)

R = numpy.column_stack([roll, pitch, yaw])
print("Rotation matrix: ", R)

# Need t_mug_cam
t_mug_model = numpy.eye(4)
t_mug_model[:3, :3] = R
t_mug_model[:3, 3] = mesh.centroid
trans_init = t_mug_cam @ numpy.linalg.inv(t_mug_model)


# point to point registration
mug_point_cloud_model = mesh.sample_points_uniformly(numble_of_points=100000)
source = o3d.io.read_point_cloud(mug_point_cloud_model)
target = o3d.io.read_point_cloud(mug_point_cloud_cam)

print("Initial alignment")
threshold = 0.02
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)


# draw inital alignment
draw_registration_result(source, target, reg_p2p.transformation)

# Get the SDF value
# trimesh.proximity.signed_distance(mesh, position)

# def query_SDF(q):
#     """
#     Query SDF value from mug STL file.

#     parameters
#     ----------
#     q: numpy.ndarray
#         A 4x4 matrix representing the current gripper pose in the model frame.
#         All translational units in this matrix are in meters ???
#     """
#     x = q[0][3]
#     y = q[1][3]
#     z = q[2][3]
#     point = [x,y,z]
#     value = trimesh.proximity.signed_distance(point)
#     return value
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
# print("If watertight:", mesh.is_watertight)
# print("Raw extents (inch):", mesh.extents)
# print("Raw bounds (inch):", mesh.bounds)

# Suppose the unit of STL is inch
# Convert inch into meter
mesh.apply_scale(0.0254)
print("scaled extents (m):", mesh.extents)
print("scaled bounds (m):", mesh.bounds)
# print("Edges: ", mesh.edges)
# print("Facets: ", mesh.facets)
# print("Centroid: ", mesh.centroid)

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
# numpy.savez(
#     "mug_sdf.numpyz",
#     sdf=sdf_grid,
#     xs=xs,
#     ys=ys,
#     zs=zs,
#     voxel_size=voxel_size
# )

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

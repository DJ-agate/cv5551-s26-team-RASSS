import numpy as np

import trimesh
import mesh_to_sdf

mesh = trimesh.load_mesh("Mug_w_tags.stl")
print(mesh.extents)
mesh.apply_scale(0.02)
print(mesh.extents)


voxels, grads = mesh_to_sdf.mesh_to_voxels(mesh, voxel_resolution=200, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False, check_result=False, return_gradients=True)



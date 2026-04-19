import numpy
import open3d as o3d
pcd = o3d.io.read_point_cloud("cube_point_cloud.pcd")
mug_pc = numpy.asarray(pcd.points).astype(numpy.float32)
numpy.save("mug_pc.npy", mug_pc)

# # Launch Contact-GraspNet inference
# # Modified contact_graspnet/contact_graspnet/data.py: added cam_K=None
# python contact_graspnet/inference.py --np_path=/home/rob/RASSS/mug_pc.npz \
#                                      --forward_passes=5 \
#                                      --z_range=[0.2,1.1]
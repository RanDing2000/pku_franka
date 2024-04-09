## obtain the input of the model
## read all the npy files in the folder /home/hyperpanda/Haoran/scenes/2024-04-04-21-28/single_scene
import os
import numpy as np
import open3d as o3d
import torch
from pysdf import SDF

# def alpha_shape_mesh_reconstruct(np_points, alpha=0.5, mesh_fix=False, visualize=False):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np_points)
    
#     tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha, tetra_mesh, pt_map
#     )

#     if mesh_fix:
#         mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
#         mf.repair()

#         mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mf.mesh[0]), 
#                                          triangles=o3d.utility.Vector3iVector(mf.mesh[1]))

#     if visualize:
#         if mesh_fix:
#             plt = pv.Plotter()
#             point_cloud = pv.PolyData(np_points)
#             plt.add_mesh(point_cloud, color="k", point_size=10)
#             plt.add_mesh(mesh)
#             plt.add_title("Alpha Shape Reconstruction")
#             plt.show()
#         else:
#             o3d.visualization.draw_geometries([pcd, mesh], title="Alpha Shape Reconstruction")

#     return mesh

# def point_cloud_to_tsdf(points):
#     ## points -> mesh -> sdf -> tsdf
#     mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
#     x, y, z = torch.meshgrid(torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40))
    
#     ## change torch to numpy
#     x, y, z = x.numpy(), y.numpy(), z.numpy()
#     pos = torch.stack((x, y, z), dim=-1).float() # (1, 40, 40, 40, 3)
#     pos = pos.numpy()
#     pos = pos.view(-1, 3)
#     f = SDF(mesh.vertices, mesh.triangles)
#     sdf = f(pos)
#     sdf_reshaped = sdf.reshape(40, 40, 40)
#     sdf_trunc = 4 * (0.3/40)

#     mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc) 

#     tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
#     # tsdf = tsdf[mask]
#     tsdf[mask] = 0
#     return tsdf

## save numpy as ply
def save_ply(points, filename):
    header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(header.format(len(points)))
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

path = '/home/hyperpanda/Haoran/scenes/2024-04-04-21-28/single_scene'
files = os.listdir(path)
initial_points = []
for file in files:
    if file.endswith('.npy'):
        ## list of numpy arrays
        initial_points.append(np.load(os.path.join(path, file))/10.0)

depth_npz = np.load('/home/hyperpanda/Haoran/scenes/2024-04-04-21-24/single_scene/raw.npz')

## concatee the list of numpy arrays
initial_points = np.concatenate(initial_points, axis=0)

save_ply(initial_points, 'initial_points.ply')

transformation_matrix = np.array([[ 0.99801218, -0.0359675 , -0.05174974,  0.1546653 ],
       [ 0.03723417, -0.32598377,  0.94464184, -0.55582304],
       [-0.05084598, -0.94469091, -0.32399655,  0.42365913],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

for i in range(initial_points.shape[0]):
    initial_points[i] = np.dot(transformation_matrix[:3, :3], initial_points[i]) + transformation_matrix[:3, 3]


print("after transformation")
initial_points = np.clip(initial_points, 0, 0.3)
save_ply(initial_points, 'initial_points_transforme_within_bound.ply')


## how to create a cube ranging from -1 to 1
cube = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
])

## save cube
save_ply(cube, 'cube.ply')  
print("plane")

import numpy as np
import open3d as o3d
import json
import re
from utils_scene import *


def bound_points(point_cloud):
    # Clip the point cloud values to the range (0, 0.3)
    bounded_point_cloud = np.clip(point_cloud, 0, 0.3)
    return bounded_point_cloud

def get_grid(tsdf_volume,resolution=40):
    shape = (1, resolution, resolution, resolution)
    tsdf_grid = np.zeros(shape, dtype=np.float32)
    voxel_grid = tsdf_volume.extract_voxel_grid()
    voxels = voxel_grid.get_voxels()
    for voxel in voxels:
        i, j, k = voxel.grid_index
        tsdf_grid[0, i, j, k] = voxel.color[0]
    return tsdf_grid


def try_get_grid(tsdf_volume,resolution=40):
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        try:
            tsdf_grid = get_grid(tsdf_volume,resolution)
            tsdf_to_ply(tsdf_grid[0], f'{clutter_scene_path}/targ_tsdf.ply')
    
            print("end")
            return tsdf_grid
        except Exception as e:
            print(f"Attempt {attempts+1}/{max_attempts} failed:", e)
            attempts += 1
    raise RuntimeError(f"Failed to get grid after {max_attempts} attempts")


if __name__=="__main__":
    ## ---------------------------------------- ##
    # firstly, do the single scene
    ## ---------------------------------------- ##
    ## randomly name the scene, 8 characters
    # scene_name = time.strftime("%Y-%m-%d-%H-%M", time.localtime())  ## year-month-day-hour-minute
    
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")

    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    clutter_scene_path  = os.path.join(save_dir, 'clutter_scene')

    T_cam2plane = np.load(f'{save_dir}/cam2plane_transformation.npy')

    # extrinsic_inv = np.linalg.inv(extrinsic)
    T_plane2cam = inverse_extrinsics(T_cam2plane)

    # Regular expression pattern to match "object_<number>.npy"
    # pattern = re.compile(r'object_\d+\.npy$')
    pattern = re.compile(r'object_(0|2)\.npy$')

    # Search for files matching the pattern in the directory
    matching_files = [f for f in os.listdir(clutter_scene_path) if pattern.match(f)]

    arrays = []
    for file in matching_files:
        arrays.append(np.load(os.path.join(clutter_scene_path, file)))
    
    point_scene_cam = np.concatenate(arrays, axis=0) 
    
    ## load ply and convert to numpy
    #point_targ_cam = np.load('/home/hyperpanda/Haoran/object_3.ply')
    # point_targ_cam = np.asarray(o3d.io.read_point_cloud('/home/hyperpanda/Haoran/object_3_cropped.ply').points) / 10.0
    point_targ_cam = np.load(f'{save_dir}/clutter_scene/object_2.npy')

    # np.save(f'{clutter_scene_path}/target_pointcloud_cam.npy', point_targ_cam)

    point_scene_plane = transform_point_cloud(point_scene_cam, T_cam2plane)
    point_targ_plane = transform_point_cloud(point_targ_cam, T_cam2plane)

    point_scene_plane = bound_points(point_scene_plane)
    point_targ_plane = bound_points(point_targ_plane)

    point_targ_cam = transform_point_cloud(point_targ_plane, T_plane2cam)

    focal_length = [913.576, 912.938]
    principal_point = [628.32, 360.564]
    intrinsics = np.array([
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1]
    ])

    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=1280,
        height=720,
        fx=913.576,
        fy=912.938,
        cx=628.32,
        cy=360.564,
    )

    ## remove nan values
    # point_np = point_np[~np.isnan(point_np).any(axis=1)]
    point_scene_cam = point_scene_cam[~np.isnan(point_scene_cam).any(axis=1)]
    # depth_image = pointcloud_to_depthmap(point_scene_cam, intrinsics, (1280, 720))
    # # save_depth_image(depth_image, f'{clutter_scene_path}/clutter_scene_depth_image.png')

    # ---------------------------------------- #
    depth_image = pointcloud_to_depthmap(point_targ_cam, intrinsics, (1280, 720))
    # depth_image = pointcloud_to_depthmap(point_scene_cam, intrinsics, (1280, 720))

    tsdfvolume = o3d.pipelines.integration.UniformTSDFVolume(
                length=0.3,
                resolution=40,
                sdf_trunc=1.2/40,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            )

    # # tsdf
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.empty_like(depth_image).astype(np.uint16)),
        o3d.geometry.Image(depth_image.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )

    ## get depth from rgbd
    depth = np.asarray(rgbd.depth)
    tsdfvolume.integrate(rgbd, intrinsics_o3d, T_plane2cam)
    tsdf_grid = try_get_grid(tsdfvolume)

    
# Assume `rgbd_images` is a list of RGBDImage objects and `camera_intrinsics` is an o3d.camera.PinholeCameraIntrinsic object
# Initialize ScalableTSDFVolume
tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_size=0.05,
    sdf_trunc=0.1,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
)

tsdf_volume.integrate(rgbd, intrinsics_o3d, T_plane2cam)

# Extract point cloud or mesh
extracted_mesh = tsdf_volume.extract_triangle_mesh()
extracted_mesh.compute_vertex_normals()
extracted_point_cloud = tsdf_volume.extract_point_cloud()

# Optionally, convert to a voxel grid
def point_cloud_to_voxel_grid(point_cloud, voxel_size):
    points = np.asarray(point_cloud.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    dims = ((max_bound - min_bound) // voxel_size).astype(int) + 1
    voxel_grid = np.zeros(dims, dtype=bool)

    indices = ((points - min_bound) // voxel_size).astype(int)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return voxel_grid

voxel_size = 0.05  # Define the voxel size
voxel_grid = point_cloud_to_voxel_grid(extracted_point_cloud, voxel_size)
import numpy as np
import open3d as o3d
import json
import re
from utils_scene import *


def bound_points(point_cloud):
    # Clip the point cloud values to the range (0, 0.3)
    bounded_point_cloud = np.clip(point_cloud, 0, 0.3)
    return bounded_point_cloud

# # A, B: a point cloud
# def find_closest_point(A, B):
#     # Calculate the squared Euclidean distances
#     distances_squared = np.sum((B - A)**2, axis=1)
#     # Find the index of the minimum distance
#     min_index = np.argmin(distances_squared)
#     # Return the closest point
#     return B[min_index]


def get_grid(tsdf_volume,resolution=40):
    shape = (1, resolution, resolution, resolution)
    tsdf_grid = np.zeros(shape, dtype=np.float32)
    voxel_grid = tsdf_volume.extract_voxel_grid()
    voxels = voxel_grid.get_voxels()
    for voxel in voxels:
        i, j, k = voxel.grid_index
        tsdf_grid[0, i, j, k] = voxel.color[0]
    return tsdf_grid


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

        # ## intrinsics is a matrix
    # intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
    #     width=1280,
    #     height=720,
    #     fx=913.576,
    #     fy=912.938,
    #     cx=628.32,
    #     cy=360.564,
    # )

    # extrinsic = np.eye(4)  # Identity matrix
    # T_cam2plane = np.array([[ 0.99763501, -0.04145525, -0.05482554,  0.15870768],
    #     [ 0.03871833, -0.32011408,  0.94658748, -0.55869447],
    #     [-0.05679145, -0.94647157, -0.31775194,  0.41806738],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]])

    T_cam2plane = np.load(f'{save_dir}/cam2plane_transformation.npy')

    # extrinsic_inv = np.linalg.inv(extrinsic)
    T_plane2cam = inverse_extrinsics(T_cam2plane)

    # Regular expression pattern to match "object_<number>.npy"
    pattern = re.compile(r'object_\d+\.npy$')

    # Search for files matching the pattern in the directory
    matching_files = [f for f in os.listdir(clutter_scene_path) if pattern.match(f)]

    arrays = []
    for file in matching_files:
        arrays.append(np.load(os.path.join(clutter_scene_path, file)))
    
    point_scene_cam = np.concatenate(arrays, axis=0) 
    # np.save()
    np.save(f'{clutter_scene_path}/scene_pointcloud_cam.npy', point_scene_cam)

    point_targ_cam = np.load(f'{save_dir}/clutter_scene/object_0.npy') 

    # np.save(f'{clutter_scene_path}/target_pointcloud_cam.npy', point_targ_cam)

    point_scene_plane = transform_point_cloud(point_scene_cam, T_cam2plane)
    point_targ_plane = transform_point_cloud(point_targ_cam, T_cam2plane)

    point_scene_plane = bound_points(point_scene_plane)
    point_targ_plane = bound_points(point_targ_plane)

    point_targ_cam = transform_point_cloud(point_targ_plane, T_plane2cam)
    np.save(f'{clutter_scene_path}/target_pointcloud_cam.npy', point_targ_cam)
    

    save_pointcloud_to_ply(point_scene_plane, f'{clutter_scene_path}/scene_pointcloud.ply')

    np.save(f'{clutter_scene_path}/scene_pointcloud.npy', point_scene_plane)
    save_pointcloud_to_ply(point_targ_plane, f'{clutter_scene_path}/target_pointcloud.ply')
    np.save(f'{clutter_scene_path}/target_pointcloud.npy', point_targ_plane)

    # path = '/usr/stud/dira/GraspInClutter/grasping/initial_points.ply'
    # pcd = o3d.io.read_point_cloud(path)
    # point_np = np.asarray(pcd.points)

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
    depth_image = pointcloud_to_depthmap(point_scene_cam, intrinsics, (1280, 720))
    save_depth_image(depth_image, f'{clutter_scene_path}/clutter_scene_depth_image.png')

    tsdfvolume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.3 / 40,  # 体素的物理大小，由体积长度除以分辨率得到
    sdf_trunc=1.2 / 40,  # 截断距离，与之前相同
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,  # 指定不使用颜色信息
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
    # save_depth_image(depth, '/usr/stud/dira/GraspInClutter/grasping/depth_image_rgbd.png')
    save_depth_image(depth, f'{clutter_scene_path}/clutter_scene_depth_image_rgbd.png')

    tsdfvolume.integrate(rgbd, intrinsics_o3d, T_plane2cam)
    # tsdfvolume.integrate(rgbd, intrinsics_o3d, T_cam2plane)

    save_pointcloud_to_ply(np.asarray(tsdfvolume.extract_point_cloud().points), f'{clutter_scene_path}/clutter_scene_tsdf_points.ply')

    tsdf_grid = get_grid(tsdfvolume)
    np.save(f'{clutter_scene_path}/clutter_scene_tsdf_grid.npy', tsdf_grid)

    tsdf_to_ply(tsdf_grid[0], f'{clutter_scene_path}/clutter_scene_tsdf.ply')


    # ---------------------------------------- #
    depth_image = pointcloud_to_depthmap(point_targ_cam, intrinsics, (1280, 720))

    ## save depth_image

    
    save_depth_image(depth_image, f'{clutter_scene_path}/single_scene_depth_image.png')

    tsdfvolume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.3 / 40,  # 体素的物理大小，由体积长度除以分辨率得到
    sdf_trunc=1.2 / 40,  # 截断距离，与之前相同
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,  # 指定不使用颜色信息
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
    # save_depth_image(depth, '/usr/stud/dira/GraspInClutter/grasping/depth_image_rgbd.png')
    # save_depth_image(depth, f'{clutter_scene_path}/clutter_scene_depth_image_rgbd.png')

    tsdfvolume.integrate(rgbd, intrinsics_o3d, T_plane2cam)
    # tsdfvolume.integrate(rgbd, intrinsics_o3d, T_cam2plane)

    # save_pointcloud_to_ply(np.asarray(tsdfvolume.extract_point_cloud().points), f'{clutter_scene_path}/clutter_scene_tsdf_points.ply')

    tsdf_grid = get_grid(tsdfvolume)
    np.save(f'{clutter_scene_path}/targ_tsdf_grid.npy', tsdf_grid)

    tsdf_to_ply(tsdf_grid[0], f'{clutter_scene_path}/targ_tsdf.ply')
    
    print("end")
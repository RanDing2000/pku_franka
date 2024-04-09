import numpy as np
import open3d as o3d
import json
import re
from utils_scene import *
from pysdf import SDF
import torch

def bound_points(point_cloud):
    # Clip the point cloud values to the range (0, 0.3)
    bounded_point_cloud = np.clip(point_cloud, 0, 0.3)
    return bounded_point_cloud

def get_grid(tsdf_volume,resolution=40):
    # shape = (1, resolution, resolution, resolution)
    # tsdf_grid = np.zeros(shape, dtype=np.float32)
    vertices = np.asarray(tsdf_volume.extract_triangle_mesh().vertices)
    triangle_mesh = np.asarray(tsdf_volume.extract_triangle_mesh().triangles)
    
    x, y, z = torch.meshgrid(torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40))
    pos = torch.stack((x, y, z), dim=-1).float() # (1, 40, 40, 40, 3)
    pos = pos.view(-1, 3)

    f = SDF(vertices, triangle_mesh)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(40,40,40)
    sdf_trunc = 4 * (0.3/40)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc) 
    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    tsdf[mask] = 0
    return tsdf


if __name__=="__main__":
    ## ---------------------------------------- ##
    # firstly, do the single scene
    ## ---------------------------------------- ##
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
    pattern = re.compile(r'object_\d+\.npy$')

    # Search for files matching the pattern in the directory
    matching_files = [f for f in os.listdir(clutter_scene_path) if pattern.match(f)]
    # matching_files = ["object_0_cropped.npy"]

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

    intrinsics = np.load(os.path.join(save_dir, "intrinsics.npy"), allow_pickle=True)
    focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
    principal_point = [intrinsics[0, 2], intrinsics[1, 2]]
    
    width, height = 1280, 720

    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=focal_length[0],
        fy=focal_length[1],
        cx=principal_point[0],
        cy=principal_point[1],
    )


    ## remove nan values
    # point_np = point_np[~np.isnan(point_np).any(axis=1)]
    point_scene_cam = point_scene_cam[~np.isnan(point_scene_cam).any(axis=1)]
    depth_image = pointcloud_to_depthmap(point_scene_cam, intrinsics, (width, height))
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
    # save_depth_image(depth, f'{clutter_scene_path}/clutter_scene_depth_image_rgbd.png')

    tsdfvolume.integrate(rgbd, intrinsics_o3d, T_plane2cam)
    # tsdfvolume.integrate(rgbd, intrinsics_o3d, T_cam2plane)

    save_pointcloud_to_ply(np.asarray(tsdfvolume.extract_point_cloud().points), f'{clutter_scene_path}/clutter_scene_tsdf_points.ply')

    tsdf_grid = get_grid(tsdfvolume)
    np.save(f'{clutter_scene_path}/clutter_scene_tsdf_grid.npy', tsdf_grid)

    tsdf_to_ply(tsdf_grid, f'{clutter_scene_path}/clutter_scene_tsdf.ply')


    # ---------------------------------------- #
    depth_image = pointcloud_to_depthmap(point_targ_cam, intrinsics, (width, height))

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

    tsdf_to_ply(tsdf_grid, f'{clutter_scene_path}/targ_tsdf.ply')
    
    print("Done!")
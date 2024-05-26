import os
import numpy as np
import open3d as o3d
import json
import re
from utils_scene import *
from pysdf import SDF
import torch

def bound_points(point_cloud):
    # bound the point cloud values to the following range
    lower = np.array([0.02, 0.02, 0.07])
    upper = np.array([0.28, 0.28, 0.30])
    mask = np.all((point_cloud >= lower) & (point_cloud <= upper), axis=1)
    # Filter the points using the mask
    filtered_point_cloud = point_cloud[mask]
    return filtered_point_cloud


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

    ## convert to numpy
    # tsdf = tsdf.cpu().numpy()

    return tsdf


if __name__=="__main__":
    ## ---------------------------------------- ##
    # firstly, do the single scene
    ## ---------------------------------------- ##
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root", type=str, default="/Users/ziyuan/Desktop/Github/pku")  # "/home/hyperpanda/Haoran")

    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    clutter_scene_path  = os.path.join(save_dir, 'clutter_scene')
    single_scene_path = os.path.join(save_dir, 'single_scene')

    T_cam2plane = np.load(f'{save_dir}/cam2plane_transformation.npy') # /Users/ziyuan/Desktop/Github/pku/scenes/2024-04-08-23-35-rbmwtgignfhapyfm/cam2plane_transformation.npy

    # extrinsic_inv = np.linalg.inv(extrinsic)
    T_plane2cam = inverse_extrinsics(T_cam2plane)

    # Regular expression pattern to match "object_<number>.npy"
    pattern = re.compile(r'object_\d+\.npy$')

    # Search for files matching the pattern in the directory
    matching_files = [f for f in os.listdir(clutter_scene_path) if pattern.match(f)]
    sorted_files = sorted(matching_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    no_targ_arrays = []
    all_obj_arrays = []
    for file in sorted_files:
        if (file != 'object_0.npy') and (file != 'object_1.npy'): # (file != sorted_files[-1]):
            no_targ_arrays.append(np.load(os.path.join(clutter_scene_path, file)))
        all_obj_arrays.append(np.load(os.path.join(clutter_scene_path, file)))
    
    if len(all_obj_arrays) == 0:
        raise Exception("No objects in the scene!")
    pc_scene_cam = np.concatenate(all_obj_arrays, axis=0)
    
    pc_targ_cam = np.load(f'{save_dir}/clutter_scene/object_0.npy')
    pc_targ_plane = transform_point_cloud(pc_targ_cam, T_cam2plane)
    pc_targ_plane = bound_points(pc_targ_plane)
    
    if len(no_targ_arrays) > 0:
        pc_scene_no_targ_cam = np.concatenate(no_targ_arrays, axis=0)
        pc_scene_no_targ_plane = transform_point_cloud(pc_scene_no_targ_cam, T_cam2plane)
        save_pointcloud_to_ply(pc_scene_no_targ_plane, f'{clutter_scene_path}/pc_scene_no_targ_plane_before_crop.ply')
        save_pointcloud_to_ply(pc_targ_plane, f'{clutter_scene_path}/pc_targ_plane_before_crop.ply')
        pc_scene_no_targ_plane = bound_points(pc_scene_no_targ_plane)
        save_pointcloud_to_ply(pc_scene_no_targ_plane, f'{clutter_scene_path}/pc_scene_no_targ_plane.ply')
        save_pointcloud_to_ply(pc_targ_plane, f'{clutter_scene_path}/pc_targ_plane.ply')
    else:
        pc_scene_no_targ_plane = np.array([])
        pc_scene_no_targ_cam = np.array([])
    
    np.save(f'{clutter_scene_path}/pc_scene_no_targ_cam.npy', pc_scene_no_targ_cam)
    
    pc_targ_cam = transform_point_cloud(pc_targ_plane, T_plane2cam)
    np.save(f'{clutter_scene_path}/pc_targ_cam.npy', pc_targ_cam)
    np.save(f'{clutter_scene_path}/pc_scene_no_targ_plane.npy', pc_scene_no_targ_plane)
    np.save(f'{clutter_scene_path}/pc_targ_plane.npy', pc_targ_plane)

    intrinsics = np.load(f"{save_dir}/intrinsics.npy", allow_pickle=True)
    
    focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
    principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=1280,
        height=720,
        fx=focal_length[0],
        fy=focal_length[1],
        cx=principal_point[0],
        cy=principal_point[1],
    )

    ## remove nan values
    pc_scene_cam = pc_scene_cam[~np.isnan(pc_scene_cam).any(axis=1)]
    depth_image = pointcloud_to_depthmap(pc_scene_cam, intrinsics, (1280, 720))
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

    tsdf_to_ply(tsdf_grid, f'{clutter_scene_path}/clutter_scene_tsdf.ply')


    # ------------------ single scene (unoccluded) ---------------------- #
    
    depth_image = pointcloud_to_depthmap(pc_targ_cam, intrinsics, (1280, 720))
    save_depth_image(depth_image, f'{single_scene_path}/single_scene_depth_image.png')

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
    tsdf_grid = get_grid(tsdfvolume)
    np.save(f'{single_scene_path}/targ_tsdf_grid.npy', tsdf_grid)

    tsdf_to_ply(tsdf_grid, f'{single_scene_path}/targ_tsdf.ply')
    
    print("end")

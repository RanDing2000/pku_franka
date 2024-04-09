import argparse
from copy import deepcopy
import os
import time

import cv2
import numpy as np
import sys
import random
import string

import pyrealsense2 as rs



import sys
sys.path.append("/home/hyperpanda/segment-anything")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import open3d as o3d

def inverse_extrinsics(E):
    """
    计算外参矩阵的逆矩阵。
    
    参数:
    E -- 外参矩阵, 尺寸为4x4。
    
    返回:
    E_inv -- 外参矩阵的逆矩阵。
    """
    R = E[0:3, 0:3]  # 提取旋转矩阵
    t = E[0:3, 3]    # 提取平移向量
    
    R_inv = R.T  # 计算旋转矩阵的逆（即转置）
    t_inv = -np.dot(R_inv, t)  # 计算逆平移向量
    
    E_inv = np.identity(4)  # 创建一个4x4的单位矩阵
    E_inv[0:3, 0:3] = R_inv  # 将旋转部分填入
    E_inv[0:3, 3] = t_inv    # 将平移部分填入
    
    return E_inv

def compute_intersection(a, b):
    a_tuples = set(map(tuple, a))
    b_tuples = set(map(tuple, b))
    intersection_tuples = np.array(list(a_tuples & b_tuples))
    return np.array(intersection_tuples)

def create_onclick_handler(color_image, anchor):
    def onclick(event):
        if event.button == 1:  # 左键单击
            x, y = int(event.xdata), int(event.ydata)
            pixel_value = color_image[y, x]
            print(f"点击位置 ({x}, {y}) 的像素值为 {pixel_value}")
            anchor.append([x, y])
    return onclick

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def depth2pcd(depth,intrinsics):
    pcd=np.zeros((depth.shape[0],depth.shape[1],3))
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            z=depth[i,j]
            x=(j-intrinsics.ppx)*z/intrinsics.fx
            y=(i-intrinsics.ppy)*z/intrinsics.fy
            pcd[i,j]=np.array([x,y,z])
    return pcd

# def onclick(event):
#     if event.button == 1:  # 左键单击
#         x, y = int(event.xdata), int(event.ydata)
#         pixel_value = color_image[y, x]
#         print(f"点击位置 ({x}, {y}) 的像素值为 {pixel_value}")
#         anchor.append([x, y])


def process_one_round(save_dir, sam):
    # print("-------------------load sam -------------------")
    predictor = SamPredictor(sam)
    
    print("-------------------start camera -------------------")
    #Take image and depth data from realsense
    pipeline = rs.pipeline()
    config = rs.config()
    # Start streaming
    pipeline.start(config)
    # when startig the camera, the light is so dim, and it needs time to adjust the lighting
    time.sleep(2)
    
    
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    # intrinsics_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],[0, color_intrinsics.fy, color_intrinsics.ppy],[0, 0, 1]])

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data()) 

    color_image = np.uint8(np.asanyarray(color_frame.get_data()))
    color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)

    height, width  = depth_image.shape

    # Stop streaming
    pipeline.stop()
    
    # save depth and color as npz
    np.savez_compressed(os.path.join(save_dir, "raw.npz"), depth=depth_image, color=color_image)
    
    print("color_image",color_image.shape)

    # depth_image = cv2.resize(depth_image, (width//2, height//2))
    # color_image = cv2.resize(color_image, (width//2, height//2))
    
    depth_pc=depth2pcd(depth_image,color_intrinsics)

    
    
    anchor=[]


    onclick = create_onclick_handler(color_image, anchor)


    fig, ax = plt.subplots(figsize=(10, 10))
    color_image_show = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    ax.imshow(color_image_show)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # cv2.imshow("rgb",rgb)
    # cv2.waitKey(0)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(color_image)
    # # plt.connect('button_press_event', onclick(color_image))
    # # plt.show()
    # onclick = create_onclick_handler(color_image)
    # # fig.canvas.mpl_connect('button_press_event', onclick_handler)
    # plt.connect('button_press_event', onclick)
    
    # plt.show()
    
    predictor.set_image(color_image)
    
    
    whole_points=[]
    whole_colors=[]
    object_points=[]
    object_pcds=[]
    object_colors=[]
    label_colors=[]
    segmentation_map = np.zeros_like(color_image, dtype=np.uint8)
    
    # Instance segmentation
    for id,object in enumerate(anchor):
        input_point=np.array([object])
        input_label=np.array([1])
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        score=scores[np.argmax(scores)]
        mask=masks[np.argmax(scores)]
        plt.figure(figsize=(10, 10))
        color_image_show = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        plt.imshow(color_image_show)
        show_mask(mask, plt.gca())
        plt.title(f"Mask 0, Score: {score:.3f}", fontsize=18)
        plt.axis('on')
        plt.show()
        
        object_point=[]
        object_color=[]
        for i in range(color_image.shape[0]):
            for j in range(color_image.shape[1]):
                if mask[i,j]:
                    object_point.append(depth_pc[i,j])
                    object_color.append(color_image[i,j])
                    segmentation_map[i, j] = id + 1 # background is 0, target is 1
        
        object_point=np.array(object_point)
        object_color=np.array(object_color)
        object_point=object_point.astype(np.float32)
        object_color=object_color.astype(np.float32)
        object_point/=1000
        object_color/=255
        
        # print("object_point",object_point)
        # print("object_color",object_color)
        
        label_color=np.random.uniform(0,1,(1,3))
        label_color=np.repeat(label_color,object_point.shape[0],axis=0)
        object_points.append(object_point)
        object_colors.append(object_color)
        
        object_pcd=o3d.geometry.PointCloud()
        object_pcd.points=o3d.utility.Vector3dVector(object_point)
        object_pcd.colors=o3d.utility.Vector3dVector(object_color)
        
        label_colors.append(label_color)
        
        object_pcds.append(object_pcd)
        
        
    whole_points=np.concatenate(object_points,axis=0)
    whole_colors=np.concatenate(object_colors,axis=0)
    label_colors=np.concatenate(label_colors,axis=0)
    
    whole_pcd=o3d.geometry.PointCloud()
    whole_pcd.points=o3d.utility.Vector3dVector(whole_points)
    whole_pcd.colors=o3d.utility.Vector3dVector(whole_colors)
    
    whole_label_pcd=o3d.geometry.PointCloud()
    whole_label_pcd.points=o3d.utility.Vector3dVector(whole_points)
    whole_label_pcd.colors=o3d.utility.Vector3dVector(label_colors)
    
    np.save(os.path.join(save_dir, "segmentation_map.npy"), segmentation_map)
    o3d.visualization.draw_geometries([whole_pcd])
    o3d.io.write_point_cloud(os.path.join(save_dir,"whole_pcd.ply"),whole_pcd,write_ascii=True)
    o3d.visualization.draw_geometries([whole_label_pcd])
    o3d.io.write_point_cloud(os.path.join(save_dir,"whole_label.ply"),whole_label_pcd,write_ascii=True)
    for i, pcd in enumerate(object_pcds):
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(save_dir,f"object_{i}.ply"),pcd,write_ascii=True)
        ## change to numpy array
        ## save as npy
        np.save(os.path.join(save_dir,f"object_{i}.npy"),np.array(object_points[i]))

def save_pointcloud_to_ply(point_cloud, filename):
    """
    Save a point cloud to a PLY file.
    
    Parameters:
    - point_cloud: Nx3 numpy array of 3D points.
    - filename: The output PLY file.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the point cloud coordinates
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Save the point cloud to file
    o3d.io.write_point_cloud(filename, pcd)

def pointcloud_to_depthmap(point_cloud, intrinsic_matrix, image_size):
    """
    Project a point cloud onto a depth map.
    
    Parameters:
    - point_cloud: Nx3 numpy array of 3D points.
    - intrinsic_matrix: 3x3 numpy array representing the camera intrinsic matrix.
    - image_size: tuple (width, height) representing the size of the depth map.
    
    Returns:
    - depth_map: 2D numpy array with the same aspect ratio as the image_size, containing depth values.
    """
    # Initialize the depth map with zeros (or a very large value if handling occlusions)
    depth_map = np.zeros((image_size[1], image_size[0]))

    a_u = intrinsic_matrix[0, 0]
    a_v = intrinsic_matrix[1, 1]
    u = intrinsic_matrix[0, 2]
    v = intrinsic_matrix[1, 2]

    # Project each point to the image plane
    for point in point_cloud:
        # projected_point = np.dot(intrinsic_matrix, point)
        # Convert homogeneous coordinates to 2D
        x, y, z = point.flatten()
        if z > 0:
            x1 = u + a_u * x / z
            y1 = v + a_v * y / z
            x1 = int(round(x1))
            y1 = int(round(y1))

            if 0 <= x1 < image_size[0] and 0 <= y1 < image_size[1]:
                    depth_map[y1, x1] = z
                
    return depth_map


def tsdf_to_ply(tsdf_voxels, ply_filename):
    """
    Converts TSDF voxels to a PLY file, representing occupied voxels as points,
    with coordinates normalized between 0 and 1.

    Parameters:
        tsdf_voxels (numpy.ndarray): 3D array of TSDF values.
        threshold (float): Threshold to determine occupied voxels.
        ply_filename (str): Path to the output PLY file.
    """
    def write_ply(points, filename):
        with open(filename, 'w') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write(f'element vertex {len(points)}\n')
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')
            file.write('end_header\n')
            for point in points:
                # point = point* 0.3 /40
                file.write(f'{point[0]} {point[1]} {point[2]}\n')

    # Identify occupied voxels
    # occupied_indices = np.argwhere(np.logical_and(np.abs(tsdf_voxels)  > 0.1, np.abs(tsdf_voxels) <= 0.5) )
    # occupied_indices = 
    occupied_indices = np.argwhere(np.abs(tsdf_voxels)  > 0)
    # occupied_indices = np.argwhere(np.logical_and(np.abs(tsdf_voxels)  > 0.1, np.abs(tsdf_voxels) < 0.2))

    # Normalize coordinates to 0-1
    # max_coords = np.array(tsdf_voxels.shape) - 1
    # normalized_points = occupied_indices / max_coords
    normalized_points = occupied_indices * 0.3 /40

    # Write normalized points to PLY
    write_ply(normalized_points, ply_filename)
    
def project_to_image_plane(camera_coords, intrinsics):
    """
    Projects 3D points in camera coordinates onto the 2D image plane
    using the intrinsic camera matrix.
    
    Parameters:
    - camera_coords: Nx3 numpy array of 3D points in camera coordinates.
    - intrinsics: 3x3 numpy array representing the intrinsic matrix of the camera.
    
    Returns:
    - image_points: Nx2 numpy array of 2D points on the image plane.
    - depth_map: Nx1 numpy array of depth values corresponding to the 2D points.
    """
    # Add a column of ones to the 3D points to make them homogeneous coordinates
    homog_coords = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))
    
    # Project the points onto the image plane
    projected_points = np.dot(intrinsics, homog_coords.T).T
    
    # Normalize the x and y coordinates by the z coordinate to get pixel coordinates
    image_points = projected_points[:, :2] / projected_points[:, 2:3]
    
    # The depth map is just the Z coordinate
    depth_map = camera_coords[:, 2]
    
    return image_points, depth_map

def save_numpy_as_ply(points, filename):
    """
    Saves a numpy array as a PLY file.
    
    Parameters:
    - points (numpy.ndarray): The input point cloud as an Nx3 numpy array.
    - filename (str): The output PLY file.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the point cloud coordinates
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save the point cloud to file
    o3d.io.write_point_cloud(filename, pcd)
    
def scale_point_cloud_to_range(points, target_min=-0.5, target_max=0.5):
    """
    Scales the point cloud so that the coordinates are within the target range.
    
    Parameters:
    - points (numpy.ndarray): The input point cloud as an Nx3 numpy array.
    - target_min (float): The minimum value of the target range.
    - target_max (float): The maximum value of the target range.
    
    Returns:
    - scaled_points (numpy.ndarray): The scaled point cloud.
    - translation (numpy.ndarray): The translation vector used to center the points.
    - scale (float): The scale factor applied to the point cloud.
    """
    # Center the point cloud at the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Find the maximum absolute value after centering
    max_dist = np.max(np.abs(centered_points))
    
    # Compute the scale factor to bring points within the target range
    scale_factor = (target_max - target_min) / (2 * max_dist)
    
    # Apply the scale factor
    scaled_points = centered_points * scale_factor
    
    # Return the scaled points, the translation and the scale factor
    return scaled_points, centroid, scale_factor

def save_depth_image(depth, filename):
    """
    Saves a depth image to a file.
    """
    # Normalize the depth image to the range [0, 255]
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
    # Convert the depth image to uint8  
    depth_uint8 = depth_normalized.astype(np.uint8)
    # Save the depth image to a file
    cv2.imwrite(filename, depth_uint8)


def transform_point_cloud(original_point_cloud, transformation_matrix):
    # Convert to homogeneous coordinates by adding a 1 to each point
    N = original_point_cloud.shape[0]
    homogeneous_points = np.hstack((original_point_cloud, np.ones((N, 1))))
    
    # Apply the transformation matrix to each point
    transformed_homogeneous_points = homogeneous_points @ transformation_matrix.T
    
    # Convert back to 3D coordinates by removing the homogeneous coordinate
    transformed_point_cloud = transformed_homogeneous_points[:, :3]
    return transformed_point_cloud
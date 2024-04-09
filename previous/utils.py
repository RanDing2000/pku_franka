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
    ax.imshow(color_image)
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
    
    # Instance segmentationing
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
        plt.imshow(color_image)
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
        
        
        object_point=np.array(object_point)
        object_color=np.array(object_color)
        object_point=object_point.astype(np.float32)
        object_color=object_color.astype(np.float32)
        object_point/=100
        object_color/=255
        
        print("object_point",object_point)
        print("object_color",object_color)
        
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
    
    
    
    o3d.visualization.draw_geometries([whole_pcd])
    o3d.io.write_point_cloud(os.path.join(save_dir,"whole_pcd.ply"),whole_pcd,write_ascii=True)
    o3d.visualization.draw_geometries([whole_label_pcd])
    o3d.io.write_point_cloud(os.path.join(save_dir,"whole_label.ply"),whole_label_pcd,write_ascii=True)
    for i, pcd in enumerate(object_pcds):
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(save_dir,f"object_{i}.ply"),pcd,write_ascii=True)

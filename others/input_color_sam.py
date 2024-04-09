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
from utils import *

if __name__=="__main__":
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    
    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
   
    
    image_path = "/home/hyperpanda/Haoran/demos/color_image_20240403-191942.png"
    color_image = cv2.imread(image_path)
    color_image = color_image.astype(np.uint8)
    
    print("-------------------load sam -------------------")
    sam_checkpoint = "/home/hyperpanda/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
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
    o3d.visualization.draw_geometries([whole_label_pcd])
    for i, pcd in enumerate(object_pcds):
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(save_dir,f"object_{i}.ply"),pcd,write_ascii=True)

    
    
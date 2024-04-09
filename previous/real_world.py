import argparse
from copy import deepcopy
import os

import cv2
import numpy as np
import sys

import pyk4a
import torch
from pyk4a import Config, PyK4A

import sys
sys.path.append("/home/hyperpanda/segment-anything")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import open3d as o3d
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def onclick(event):
    if event.button == 1:  # 左键单击
        x, y = int(event.xdata), int(event.ydata)
        pixel_value = transformed_rgb[y, x]
        print(f"点击位置 ({x}, {y}) 的像素值为 {pixel_value}")
        anchor.append([x,y])


if __name__=="__main__":
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir",type=str,default="/home/hyperpanda/safemanip/test_package/realworld/data21")
    parser.add_argument("--id",type=int,default=3)
    args=parser.parse_args()
    save_dir=args.save_dir
    cluster_id=args.id
    
    print("-------------------load sam -------------------")
    sam_checkpoint = "/home/hyperpanda/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    
    print("-------------------start camera -------------------")
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_3072P, depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,camera_fps=pyk4a.FPS.FPS_5,
                       synchronized_images_only=True))
    k4a.start()
    
    capture = k4a.get_capture()
    depth=capture.depth
    rgb=capture.color
    transformed_depth = capture.transformed_depth
    transformed_rgb = capture.transformed_color
    rgb=cv2.cvtColor(rgb,cv2.COLOR_BGRA2RGB)
    transformed_rgb=cv2.cvtColor(transformed_rgb,cv2.COLOR_BGRA2RGB)
    depth_pc=capture.depth_point_cloud
    transformed_depth_pc=capture.transformed_depth_point_cloud

    print("transformed_rgb",transformed_rgb.shape)


    print("depth_pc",depth_pc.shape)
    # depth_pc= depth_pc.reshape(-1,3)
    
    # pcd=o3d.geometry.PointCloud()
    # pcd.points=o3d.utility.Vector3dVector(depth_pc)
    # o3d.visualization.draw_geometries([pcd])
    
    
    anchor=[]
    # cv2.imshow("rgb",rgb)
    # cv2.waitKey(0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(transformed_rgb)
    plt.connect('button_press_event', onclick)
    plt.show()
    
    predictor.set_image(transformed_rgb)
    
    
    whole_points=[]
    whole_colors=[]
    object_points=[]
    object_pcds=[]
    object_colors=[]
    label_colors=[]
    
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
        plt.imshow(transformed_rgb)
        show_mask(mask, plt.gca())
        plt.title(f"Mask 0, Score: {score:.3f}", fontsize=18)
        plt.axis('on')
        plt.show()  
        
        object_point=[]
        object_color=[]
        for i in range(transformed_rgb.shape[0]):
            for j in range(transformed_rgb.shape[1]):
                if mask[i,j]:
                    object_point.append(depth_pc[i,j])
                    object_color.append(transformed_rgb[i,j])
        
        
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
    
    os.mkdir(os.path.join(save_dir,str(cluster_id)))
    
    path=os.path.join(save_dir,str(cluster_id))
    
    o3d.visualization.draw_geometries([whole_pcd])
    o3d.io.write_point_cloud(os.path.join(path,"whole_pcd.pcd"),whole_pcd)
    
    o3d.visualization.draw_geometries([whole_label_pcd])
    o3d.io.write_point_cloud(os.path.join(path,"whole_label_pcd.pcd"),whole_label_pcd)
    
    for i, pcd in enumerate(object_pcds):
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(path,f"object_{i}.pcd"),pcd)
        
    torch.save({"depth_pc":depth_pc,"transformed_depth_pc":transformed_depth_pc,"transformed_rgb":transformed_rgb,"anchor":anchor,"rgb":rgb,"depth":depth,"transformed_depth":transformed_depth},os.path.join(path,"data.pth"))
    
        
        
    
    
    
    
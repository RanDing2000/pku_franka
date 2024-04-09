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
    scene_name = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    
    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print("-------------------load sam -------------------")
    sam_checkpoint = "/home/hyperpanda/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    print("-------------------arrange single scene -------------------")
    input("After Finish, Press Enter to continue...")
         
    ##---------------- firstly, do the single scene ----------------##
    single_scene_dir = os.path.join(save_dir, 'single_scene')
    if not os.path.exists(single_scene_dir):
        os.makedirs(single_scene_dir, exist_ok=True)
    process_one_round(single_scene_dir, sam)
    
    ## scene arrangement
    ## put enter and continue
    print("-------------------arrange clutter scene -------------------")
    input("After Finish, Press Enter to continue...")

    ##---------------- then, do the clutter scene  ----------------##
    clutter_scene_dir = os.path.join(save_dir, 'clutter_scene')
    if not os.path.exists(clutter_scene_dir):
        os.makedirs(clutter_scene_dir, exist_ok=True)
    process_one_round(clutter_scene_dir, sam)

    ## calculate the occlusion level
    single_targ_object = np.asarray(o3d.io.read_point_cloud(os.path.join(single_scene_dir, 'object_0.ply')).points)
    clutter_targ_object = np.asarray(o3d.io.read_point_cloud(os.path.join(clutter_scene_dir, 'object_0.ply')).points)\
        
    ## calculate the intersection of two objects
    intersection = compute_intersection(single_targ_object, clutter_targ_object)
    occlusion_level = 1 -   intersection.shape[0] / single_targ_object.shape[0] 
    print(f"Occlusion level: {occlusion_level}")
    print("Done!")
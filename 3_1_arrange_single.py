import argparse
import os
import numpy as np
import sys
import json

import sys
sys.path.append("/home/hyperpanda/segment-anything")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import open3d as o3d
from utils_scene import *


if __name__=="__main__":
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    # parser.add_argument("--scene_id",type=str,default="")
    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print("-------------------load sam -------------------")
    sam_checkpoint = "/home/hyperpanda/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    print("-------------------arrange single scene -------------------")
    input("First click target. Then click board. After Finish, Press Enter to continue...")

    ##---------------- firstly, do the single scene ----------------##
    single_scene_dir = os.path.join(save_dir, 'single_scene')
    if not os.path.exists(single_scene_dir):
        os.makedirs(single_scene_dir, exist_ok=True)
    single_color_img = process_one_round(single_scene_dir, sam)
    cv2.imwrite(os.path.join(single_scene_dir, "single_color.png"), single_color_img)
    
    print("Done!")

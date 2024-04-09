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
import matplotlib.pyplot as plt
import open3d as o3d

if __name__=="__main__":
    ## ---------------------------------------- ##
    # firstly, do the single scene
    ## ---------------------------------------- ##
    ## randomly name the scene, 8 characters
    # scene_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))\
    ## year-month-day-hour-minute
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
    # cluster_id=args.id
    
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
    print("color_intrinsics: ", color_intrinsics)
import argparse
import os
import time
import json
import cv2
import numpy as np
import pyrealsense2 as rs

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
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
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

    intrinsics_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],[0, color_intrinsics.fy, color_intrinsics.ppy],[0, 0, 1]])
    # print(intrinsics_matrix)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data()) 

    color_image = np.uint8(np.asanyarray(color_frame.get_data()))
    color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
    height, width = depth_image.shape

    # save color as png
    cv2.imwrite(os.path.join(save_dir, "color.png"), color_image)
    
    # Stop streaming
    pipeline.stop()
    
    # save depth and color as npz
    raw_npz_path = os.path.join(save_dir, "raw.npz")
    np.savez_compressed(raw_npz_path, depth=depth_image, color=color_image)
    
    print(f"saved color and depth to: {raw_npz_path}")

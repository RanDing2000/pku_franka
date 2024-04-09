import argparse
import os
import time
import numpy as np
import json

if __name__=="__main__":
    ## ---------------------------------------- ##
    # firstly, do the single scene
    ## ---------------------------------------- ##
    scene_data = time.strftime("%Y-%m-%d-%H-%M", time.localtime())  ## year-month-day-hour-minute
    ## randomly name the scene, 8 characters
    scene_id = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 16))
    scene_name = scene_data + '-' + scene_id
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    
    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    config = {"scene_id": scene_name}
    with open(os.path.join(args.save_dir_root, "config.json"), "w") as config_file:
        json.dump(config, config_file)
    print(f"Saved scene id to {save_dir}/config.json")
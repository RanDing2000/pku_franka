import numpy as np
import json
import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
args=parser.parse_args()
with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
    config = json.load(config_file)
scene_id = config["scene_id"]
save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)

focal_length = [913.576, 912.938]
principal_point = [628.32, 360.564]
intrinsics = np.array([[focal_length[0], 0, principal_point[0]], [0, focal_length[1], principal_point[1]], [0, 0, 1]])
print(intrinsics)
intrinsics = np.load(os.path.join(save_dir, "intrinsics.npy"), allow_pickle=True)
print(intrinsics)


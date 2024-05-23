import os
import argparse
import numpy as np
import json
from pathlib import Path
from vgn.networks import load_network
import torch

from vgn.detection_implicit import process, select

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = np.load('/home/hyperpanda/GraspInClutter/realworld/data/demo_giga.npy')
    data = np.load(os.path.join(args.save_dir_root, 'scenes', scene_id, 'clutter_scene/clutter_scene_tsdf_grid.npy'), allow_pickle=True)
    data = np.expand_dims(data, axis=0)
    voxel_grid = np.load(os.path.join(args.save_dir_root, 'scenes', scene_id, 'clutter_scene/targ_tsdf_grid.npy'), allow_pickle=True)
    model_path = '/home/hyperpanda/GraspInClutter/checkpoints/giga_packed.pt'
    net = load_network(model_path, device, model_type='giga', shared_weights=True, add_single_supervision=False, fusion_type=None,feat_type=None, num_encoder_layers=None ) 
    net.eval()

    res = 40
    x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res), torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res), torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res))
    pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(device)
    pos = pos.view(1, res * res* res, 3)  ## pos: 1, 64000, -0.5, 0.475

    data = torch.from_numpy(data).to(device)
    qual_vol, rot_vol, width_vol = net(data, pos)

    qual_vol = qual_vol.reshape((res, res, res)).cpu().detach().numpy()
    rot_vol = rot_vol.reshape((res, res, res,4)).cpu().detach().numpy()
    width_vol = width_vol.reshape((res, res, res)).cpu().detach().numpy()

    # voxel_grid = data[0].cpu().detach().numpy()
    qual_vol, rot_vol, width_vol = process(voxel_grid, qual_vol, rot_vol, width_vol, out_th=0.1)

    grasps, scores = select(qual_vol.copy(), pos.view(res,res,res, 3).cpu(), rot_vol, width_vol, threshold= 0.8, force_detection=True, max_filter_size=4)

    for grasp in grasps:
        rotation = grasp.pose.rotation.as_matrix()
        translation = (grasp.pose.translation + 0.5) * 0.3
        
        print("rotation: ", rotation)
        print("translation: ", translation)
    print("score: ", scores)

    
    print("done")

if __name__ == "__main__":
    main()

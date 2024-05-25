import argparse
import numpy as np
# import json
from pathlib import Path
# import os
# import glob
# from vgn.detection import VGN
# from datetime import datetime
# from vgn.detection_implicit import VGNImplicit
# # from vgn.experiments import clutter_removal
# from vgn.experiments import target_sample, clutter_removal, target_sample_offline
from shape_completion.config import cfg_from_yaml_file
# from vgn.utils.misc import set_random_seed
from shape_completion import builder
from shape_completion.models.AdaPoinTr import AdaPoinTr
from vgn.networks import load_network
from utils_giga import tsdf_to_ply, point_cloud_to_tsdf, filter_and_pad_point_clouds
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = np.load('/home/hyperpanda/GraspInClutter/realworld/data/demo_ours.npz')
    # scene_no_targ_pc, targ_pc = data['scene_no_targ_pc'], data['targ_pc']
    scene_no_targ_pc = np.load('/home/hyperpanda/Haoran/scenes/2024-04-07-23-42-xpiysipqtuezjmnh/clutter_scene/scene_pointcloud.npy', allow_pickle=True)
    targ_pc = np.load('/home/hyperpanda/Haoran/scenes/2024-04-07-23-42-xpiysipqtuezjmnh/clutter_scene/target_pointcloud.npy', allow_pickle=True)
    scene_no_targ_pc, targ_pc = torch.from_numpy(scene_no_targ_pc).to(device), torch.from_numpy(targ_pc).to(device)

    # model = ''
    model_path = '/home/hyperpanda/GraspInClutter/checkpoints/afford_scene_targ_pc_with_sc_val_acc=0.9332.pt'

    net = load_network(model_path, device, model_type="afford_scene_targ_pc", shared_weights=True, add_single_supervision=False, fusion_type='transformer_concat', feat_type= 'Plane_feat', num_encoder_layers=2) 
    net.eval()

    sc_cfg = cfg_from_yaml_file("/home/hyperpanda/GraspInClutter/src/shape_completion/configs/stso/AdaPoinTr.yaml")
    sc_net = AdaPoinTr(sc_cfg.model)
    builder.load_model(sc_net, "/home/hyperpanda/GraspInClutter/checkpoints/shape_completion/ckpt-best-0301.pth")
    sc_net = sc_net.eval().to(device)

    with torch.no_grad():
        # sc_net = sc_net.to(device)
        completed_targ_pc = sc_net(targ_pc.unsqueeze(0))[1]
        completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc.squeeze().cpu().numpy())
        completed_targ_pc = filter_and_pad_point_clouds(completed_targ_pc)
        targ_completed_scene_pc = torch.cat([scene_no_targ_pc.unsqueeze(0), completed_targ_pc], dim=1)

    res = 40
    x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res), torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res), torch.linspace(start=-0.5, end=0.5 - 1.0 / res, steps=res))
    pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(device)   
    pos = pos.view(1, res * res* res, 3)  ## pos: 1, 64000, -0.5, 0.475

    # data = torch.from_numpy(data).to(device)
    qual_vol, rot_vol, width_vol = net([targ_completed_scene_pc, completed_targ_pc], pos)

    qual_vol = qual_vol.reshape((res, res, res)).cpu().detach().numpy()
    rot_vol = rot_vol.reshape((res, res, res,4)).cpu().detach().numpy()
    width_vol = width_vol.reshape((res, res, res)).cpu().detach().numpy()

    voxel_grid =completed_targ_grid

    qual_vol, rot_vol, width_vol = process(voxel_grid , qual_vol, rot_vol, width_vol, out_th=0.1)

    grasps, scores = select(qual_vol.copy(), pos.view(res,res,res, 3).cpu(), rot_vol, width_vol, threshold= 0.8, force_detection=True, max_filter_size=4)

    grasp = grasps[0]

    rotation = grasp.pose.rotation.as_matrix()
    translation = (grasp.pose.translation + 0.5) * 0.3

    print("done")

if __name__ == "__main__":
    main()
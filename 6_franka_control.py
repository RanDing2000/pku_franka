import numpy as np
import sys
import os
import json
import argparse
from pyquaternion import Quaternion
# from utils_scene import *
from utils_control import *

def find_closest_point(A, B):
    # Calculate the squared Euclidean distances
    distances_squared = np.sum((B - A)**2, axis=1)
    # Find the index of the minimum distance
    min_index = np.argmin(distances_squared)
    # Return the closest point
    return B[min_index]


def get_se3(input_string):
    """
    Input string should be four lines, where the first three lines are the rotation matrix
    and the fourth line is the translation vector.
    
    E.g.
    ```
    rotation:  [[ 0.65947986 -0.69682496 -0.28199521]
    [-0.66287058 -0.71598196  0.21902609]
    [-0.35452633  0.04248303 -0.93408044]]
    translation:  [0.195 0.105 0.145]
    ```
    
    Note: indentation is not important, b/c heading and trailing spaces are removed.
    """
    lines = input_string.strip().split('\n')

    # Extract and clean the rotation matrix
    rotation_lines = lines[:3]
    rotation_values = []
    for line in rotation_lines:
        cleaned_line = line.replace('rotation:', '').replace('[', '').replace(']', '').strip()
        rotation_values.extend([float(num) for num in cleaned_line.split()])

    # Extract and clean the translation vector
    translation_line = lines[3].replace('translation:', '').strip()
    translation_values = [float(num) for num in translation_line.replace('[', '').replace(']', '').split()]

    # Convert to numpy arrays
    grasp_rotation_plane = np.array(rotation_values).reshape(3, 3)
    grasp_translation_plane = np.array(translation_values)

    return grasp_rotation_plane, grasp_translation_plane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Given grasp pose, grasp the target.')
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    # parser.add_argument("--save_dir_root",type=str,default="/Users/ziyuan/Desktop/Github/pku")

    # parser.add_argument("--id",type=int,default=1)
    args=parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    clutter_scene_path  = os.path.join(save_dir, 'clutter_scene')
    
    intrinsics = np.load(os.path.join(save_dir, "intrinsics.npy"), allow_pickle=True)
    
    T_cam_base = np.load(os.path.join(save_dir, 'base2cam_transformation.npy')) ## eye = T (hand->eye) * hand 
    # T_base_cam = inverse_extrinsics(T_cam_base) ## hand = T (eye -> hand) * eye

    input_string = """
    rotation:  [[-0.44512886  0.88985296  0.10011002]
 [ 0.87192779  0.45617546 -0.17789288]
 [-0.20396623  0.00810345 -0.97894439]]
translation:  [0.18   0.1425 0.15  ]
    """
    
    grasp_rotation_plane, grasp_translation_plane = get_se3(input_string)
    
    ### if `loading from input_string` fails, manually copy to the following and edit
    #     grasp_rotation_plane = np.array(
    # [[ 0.31437481, -0.94314296, -0.1079344 ],
    #  [-0.85662919, -0.33084358,  0.39589009],
    #  [-0.40909035, -0.03199811, -0.91193267],]
    # )
    #     grasp_translation_plane = np.array([0.165, 0.0675, 0.2625])
    
    T_grasp_plane = np.eye(4)
    T_grasp_plane[:3, :3] = grasp_rotation_plane
    T_grasp_plane[:3, 3] = grasp_translation_plane
    T_cam_plane = np.load(f'{save_dir}/cam2plane_transformation.npy')    ## plane = T (cam -> plane) * cam
    T_plane_cam = inverse_extrinsics(T_cam_plane)   ## cam = T (plane -> cam ) * plane
    
    target_quat = rotmat_to_quaternion(T_cam_base[:3,:3] @ T_plane_cam[:3,:3] @ T_grasp_plane[:3, :3])
    grasp_translation_homo_plane = np.append(grasp_translation_plane, 1)

    tcp_point_cam = T_plane_cam @ grasp_translation_homo_plane
    targt_pc_cam = np.load(f'{clutter_scene_path}/pc_targ_cam.npy')



    contact_point_cam = find_closest_point(tcp_point_cam[:3], targt_pc_cam)

    contact_point_cam_homo = np.append(contact_point_cam,1)
    contact_point_base = T_cam_base @ contact_point_cam_homo

    pick_motion2(target_quat,contact_point_base[:3])

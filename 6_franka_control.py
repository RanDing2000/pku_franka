import numpy as np
from autolab_core import RigidTransform
import sys
sys.path.append('/home/hyperpanda/frankapy/')
from frankapy import FrankaArm
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

'''
translation: 
  x: -0.12261548208584613
  y: -0.4562687157930215
  z: 0.6744739999523481
rotation: 
  x: -0.6157378065331557
  y: 0.30202872525885305
  z: -0.3509921573226749
  w: 0.6375343976776667
x-axis: forward
y-axis: left
z-axis: upward
camera calibration offset around: [-0.11,-0.05,0.0]
'''

## [] TODO select grasp_plane
## [] TODO T_cam_base

if __name__=="__main__":
    focal_length = [913.576, 912.938]
    principal_point = [628.32, 360.564]
    intrinsics = np.array([
            [focal_length[0], 0, principal_point[0]],
            [0, focal_length[1], principal_point[1]],
            [0, 0, 1]
        ])

    # translation = np.array([-0.12261548208584613,-0.4562687157930215,0.6744739999523481])
    # rotation_quat = np.array([0.6375343976776667,-0.6157378065331557,0.30202872525885305,-0.3509921573226749]) #wxyz

    parser = argparse.ArgumentParser(description='Given grasp pose, grasp the target.')
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    args = parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)

    T_base_cam = np.load(os.path.join(save_dir, 'base2cam_transformation.npy')) ## eye = T (hand->eye) * hand 
    T_cam_base = inverse_extrinsics(T_base_cam) ## hand = T (eye -> hand) * eye
    # T_hand_eye = np.load('/home/hyperpanda/Haoran/scenes/2024-04-08-01-50-fizgcljccxxuxjpf/hand2eye_transformation.npy')
    # rotation_ quaternion_to_rotation_matrix(rotation_quat)
    fa = FrankaArm()
    grasp_rotation_plane = np.array([[-0.9916789, -0.04290288, 0.12137669],
                    [-0.04002956, 0.9988598, 0.02601396],
                    [-0.12235437, 0.02093884, -0.99226558]])
    grasp_translation_plane = np.array([0.1875, 0.06, 0.03])

    T_grasp_plane = np.eye(4)
    T_grasp_plane[:3, :3] = grasp_rotation_plane
    T_grasp_plane[:3, 3] = grasp_translation_plane
    T_cam_plane = np.load(f'{save_dir}/cam2plane_transformation.npy')    ## plane = T (cam -> plane) * cam
    #T_cam_plane = np.load('/home/hyperpanda/Haoran/scenes/2024-04-08-01-50-fizgcljccxxuxjpf/plane2camera_transformation.npy')
    T_plane_cam = inverse_extrinsics(T_cam_plane)   ## cam = T (plane -> cam ) * plane
    #T_plane_cam = T_cam_plane
    #T_hand_eye = inverse_extrinsics(T_hand_eye)
    # T_plane_cam @ T_grasp

    # target_quat = rotmat_to_quaternion(T_base_cam[:3,:3] @ T_plane_cam[:3,:3] @ T_grasp_plane[:3, :3])
    target_quat = rotmat_to_quaternion(T_cam_base[:3,:3] @ T_plane_cam[:3,:3] @ T_grasp_plane[:3, :3])
    grasp_translation_homo_plane = np.append(grasp_translation_plane, 1)
    # cam_point = T_plane_cam @ grasp_translation_homo
    # world_point = T_cam_base @ T_plane_cam @ grasp_translation_homo_plane

    tcp_point_cam = T_plane_cam @ grasp_translation_homo_plane
    targt_point_cam = np.load(f'{clutter_scene_path}/target_pointcloud_cam.npy')



    contact_point_cam = find_closest_point(tcp_point_cam, targt_point_cam)

    depth_img_cam = np.load(f'{save_dir}/clutter_scene/cam2plane_transformation.npy') 

    tcp_point_cam_image = convert_to_image_points(cam_point[:3], intrinsics)

    # contact_point_cam = tcp_point_cam

    contact_point_cam[2] = depth_image_cam[tcp_point_cam_image[0], tcp_point_cam_image[1]]

    # ? use @ instead ?
    contact_point_base = T_cam_base * contact_point_cam

    # result2 = 'The contact point at (507, 55), the gripper up 3D direction is [0, -2, -50] the gripper left 3D direction is [49, 6, 0]'
    # result = 'The contact point at (540, 325), the gripper up 3D direction is [49, -1, -1] the gripper left 3D direction is [-1, 49, -1]'


    pick_motion(target_quat, world_point[:3])
 
import numpy as np
import os
import json
import argparse
from pyquaternion import Quaternion
from utils_control import *
import re
# import trimesh
from plyfile import PlyData, PlyElement


# input_text = """
#     rotation: [[-0.47468115 -0.66354821  0.57825736]
#     [-0.52162619  0.74126749  0.42240812]
#     [-0.70893154 -0.10112502 -0.69798983]]
#     translation:  [0.165  0.105  0.0075]
#     """

input_text = """
rotation:  [[-0.73253352  0.26481322  0.62711131]
 [ 0.14181694  0.96038593 -0.2398892 ]
 [-0.66579471 -0.08679188 -0.74106988]]
translation:  [0.105 0.075 0.015]
"""


# rotation = np.array([[-0.47468115, -0.66354821, 0.57825736],
#                         [-0.52162619, 0.74126749, 0.42240812],
#                         [-0.70893154, -0.10112502, -0.69798983]])
# translation = np.array([0.165, 0.105, 0.0075])
## [] TODO select grasp_plane
## [] TODO T_cam_base
    # np.save('hand.npy', hand.vertices)

# def apply_transformation_to_points(points, transformations):
#     ## points (n, 3) numpy array, transformations (4, 4) numpy array
#     return np.dot(points, transformations[:3, :3].T) + transformations[:3, 3]

def save_numpy_to_ply(filename, numpy_array):
    # Ensure your numpy array is of dtype 'float32' and has three columns for x, y, z coordinates
    assert numpy_array.shape[1] == 3, "Array must have three columns for x, y, z coordinates"
    
    # Convert the numpy array to a structured array
    structured_array = np.array([tuple(row) for row in numpy_array], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Create a PlyElement instance from the structured array
    vertex_element = PlyElement.describe(structured_array, 'vertex')
    
    # Write to a .ply file
    PlyData([vertex_element]).write(filename)
    

def parse_text_to_arrays(input_text):
    # 使用正则表达式找到所有数字（包括负数和小数）
    numbers = re.findall(r'-?\d+\.?\d*', input_text)
    numbers = [float(num) for num in numbers]
    
    # 假设前9个数字属于旋转矩阵，后3个数字属于平移向量
    rotation_numbers = numbers[:9]
    translation_numbers = numbers[9:]
    
    # 转换为numpy数组
    rotation = np.array(rotation_numbers).reshape(3, 3)
    translation = np.array(translation_numbers)
    
    return rotation, translation



if __name__=="__main__":
    # parser=argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Given grasp pose, grasp the target.')
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")

    grasp_rotation_plane, grasp_translation_plane =  parse_text_to_arrays(input_text)
    
    print("grasp_rotation_plane", grasp_rotation_plane)
    
    print("grasp_translation_plane", grasp_translation_plane)




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
    
    # fa = FrankaArm()

    # translation = np.array([-0.12261548208584613,-0.4562687157930215,0.6744739999523481])
    # rotation_quat = np.array([0.6375343976776667,-0.6157378065331557,0.30202872525885305,-0.3509921573226749]) #wxyz

    # parser = argparse.ArgumentParser(description='Given grasp pose, grasp the target.')
    # parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    # args = parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)

    T_cam_base = np.load(os.path.join(save_dir, 'cam2base_transformation.npy')) ## eye = T (hand->eye) * hand 
    
    T_base_cam = inverse_extrinsics(T_cam_base) ## hand = T (eye -> hand) * eye
    # T_hand_eye = np.load('/home/hyperpanda/Haoran/scenes/2024-04-08-01-50-fizgcljccxxuxjpf/hand2eye_transformation.npy')
    # rotation_ quaternion_to_rotation_matrix(rotation_quat)
    #fa = FrankaArm()
    # TODO: read from giga output instead
    
    # grasp_rotation_plane = np.array([-0.43794862, -0.66847668,  0.60111557],
    #           [-0.5163313,   0.73440075,  0.44051961],
    #           [-0.73593681, -0.11744983, -0.66678523])
    # grasp_translation_plane = np.array([[0.135,      0.08999999, 0.0075]])
    # # grasp_rotation_plane = np.array([[-0.73278113,  0.46462429,  0.49714795],
    # #                                  [ 0.439807,    0.88087937, -0.17498954],
    # #                                  [-0.51923176,  0.09042012, -0.8498368 ]])
    # grasp_translation_plane = np.array([[0.03,   0.0825, 0.0225]])
    
    T_cam_plane = np.load(f'{save_dir}/cam2plane_transformation.npy')    ## plane = T (cam -> plane) * cam
    #T_cam_plane = np.load('/home/hyperpanda/Haoran/scenes/2024-04-08-01-50-fizgcljccxxuxjpf/plane2camera_transformation.npy')
    T_plane_cam = inverse_extrinsics(T_cam_plane)   ## cam = T (plane -> cam ) * plane
    #T_plane_cam = T_cam_plane
    #T_hand_eye = inverse_extrinsics(T_hand_eye)
    # T_plane_cam @ T_grasp

    # target_quat = rotmat_to_quaternion(T_base_cam[:3,:3] @ T_plane_cam[:3,:3] @ T_grasp_plane[:3, :3])
    rot_grasp_cam = T_plane_cam[:3,:3] @ grasp_rotation_plane
    rot_grasp_base = T_cam_base[:3,:3] @ rot_grasp_cam
    target_quat = rotmat_to_quaternion(rot_grasp_base)
    grasp_translation_homo_plane = np.append(grasp_translation_plane, 1)
    # cam_point = T_plane_cam @ grasp_translation_homo
    # world_point = T_cam_base @ T_plane_cam @ grasp_translation_homo_plane

    tcp_point_cam = T_plane_cam @ grasp_translation_homo_plane
    
    tcp_point_base = T_cam_base @ tcp_point_cam
    
    print("tcp_point_base", tcp_point_base)
    targt_point_cam = np.load(f'{clutter_scene_path}/target_pointcloud_cam.npy')
    save_numpy_to_ply(f'{clutter_scene_path}/target_pointcloud_cam.ply', targt_point_cam)

    # targ_point_base = apply_transformation_to_points(targt_point_cam)

    ## save targ_point_base as ply


    contact_point_cam = find_closest_point(tcp_point_cam[:3], targt_point_cam)
    print("contact_point_cam", contact_point_cam)
    save_numpy_to_ply(f'{clutter_scene_path}/contact_point_cam.ply', np.expand_dims(contact_point_cam, axis=0))
    save_numpy_to_ply(f'{clutter_scene_path}/tcp_point_cam.ply', np.expand_dims(tcp_point_cam[:3] ,axis=0))
    # depth_img_cam = np.load(f'{save_dir}/clutter_scene/cam2plane_transformation.npy') 

    # tcp_point_cam_image = convert_to_image_points(cam_point[:3], intrinsics)

    # # contact_point_cam = tcp_point_cam

    # contact_point_cam[2] = depth_image_cam[tcp_point_cam_image[0], tcp_point_cam_image[1]]

    contact_point_cam_homo = np.append(contact_point_cam,1)
    contact_point_base = T_cam_base @ contact_point_cam_homo
    
    print("T_cam_base",T_cam_base)
    print("contact point base:", contact_point_base)

    # result2 = 'The contact point at (507, 55), the gripper up 3D direction is [0, -2, -50] the gripper left 3D direction is [49, 6, 0]'
    # result = 'The contact point at (540, 325), the gripper up 3D direction is [49, -1, -1] the gripper left 3D direction is [-1, 49, -1]'


    pick_motion2(target_quat,tcp_point_base[:3])
 
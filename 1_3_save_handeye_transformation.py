import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
import json

# Setup argument parser to accept scene_id from terminal input
parser = argparse.ArgumentParser(description='Process translation and rotation into a transformation matrix.')
parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
args = parser.parse_args()

def inverse_extrinsics(E):
    """
    计算外参矩阵的逆矩阵。
    
    参数:
    E -- 外参矩阵, 尺寸为4x4。
    
    返回:
    E_inv -- 外参矩阵的逆矩阵。
    """
    R = E[0:3, 0:3]  # 提取旋转矩阵
    t = E[0:3, 3]    # 提取平移向量
    
    R_inv = R.T  # 计算旋转矩阵的逆（即转置）
    t_inv = -np.dot(R_inv, t)  # 计算逆平移向量
    
    E_inv = np.identity(4)  # 创建一个4x4的单位矩阵
    E_inv[0:3, 0:3] = R_inv  # 将旋转部分填入
    E_inv[0:3, 3] = t_inv    # 将平移部分填入
    
    return E_inv

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    q = np.array(quaternion) / np.linalg.norm(quaternion)

    w, x, y, z = q[0], q[1], q[2], q[3]

    # Compute rotation matrix elements
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                  [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                  [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])

    return R


# Assume input_string is provided in the script or loaded from a file
input_string = """
translation: 
  x: 1.0603008991349265
  y: -0.535692064544872
  z: 0.5837862333801264
rotation: 
  x: -0.679377394085064
  y: -0.3182985798358492
  z: 0.3020904872828172
  w: 0.5881102855535675
"""

# Function to parse the provided input string
def parse_input(input_str):
    sections = input_str.strip().split('\n')
    translation = {}
    rotation = {}
    current_section = None
    
    for line in sections:
        if 'translation:' in line:
            current_section = 'translation'
        elif 'rotation:' in line:
            current_section = 'rotation'
        else:
            key, value = line.strip().split(':')
            if current_section == 'translation':
                translation[key.strip()] = float(value.strip())
            elif current_section == 'rotation':
                rotation[key.strip()] = float(value.strip())
    
    return translation, rotation

# Parse the input string
translation_values, rotation_values = parse_input(input_string)

# Creating arrays for translation and rotation
translation = np.array([translation_values['x'], translation_values['y'], translation_values['z']])
rotation = np.array([rotation_values['x'], rotation_values['y'], rotation_values['z'], rotation_values['w']])

# Convert quaternion to rotation matrix
# r = R.from_quat(rotation)
# rotation_matrix = r.as_matrix()
rotation_matrix = quaternion_to_rotation_matrix(rotation)

# Create transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = translation

# Save transformation matrix to .npy file with scene_id in the filename
with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
    config = json.load(config_file)
scene_id = config["scene_id"]

filename = f'/home/hyperpanda/Haoran/scenes/{scene_id}/cam2base_transformation.npy'

# transformation_matrix = np.load(filename)
np.save(filename, transformation_matrix)



print(f"Transformation matrix saved as '{filename}'")

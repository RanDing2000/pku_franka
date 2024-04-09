import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
import json

# Setup argument parser to accept scene_id from terminal input
parser = argparse.ArgumentParser(description='Process translation and rotation into a transformation matrix.')
parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
args = parser.parse_args()

# Assume input_string is provided in the script or loaded from a file
input_string = """
translation: 
  x: 1.0800069701248212
  y: -0.5061842538039398
  z: 0.617478823738227
rotation: 
  x: -0.6226072096632805
  y: -0.42311531772752997
  z: 0.42509711618120727
  w: 0.5026192716102875
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
r = R.from_quat(rotation)
rotation_matrix = r.as_matrix()

# Create transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = translation

# Save transformation matrix to .npy file with scene_id in the filename
with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
    config = json.load(config_file)
scene_id = config["scene_id"]
filename = f'/home/hyperpanda/Haoran/scenes/{scene_id}/base2cam_transformation.npy'
np.save(filename, transformation_matrix)

print(f"Transformation matrix saved as '{filename}'")

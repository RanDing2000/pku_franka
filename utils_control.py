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

fa = FrankaArm()

## A is a point, B is a point cloud 
## find a point C in the B that has minimum distance with A
def find_closest_point(A, B):
    # Calculate the squared Euclidean distances
    distances_squared = np.sum((B - A)**2, axis=1)
    # Find the index of the minimum distance
    min_index = np.argmin(distances_squared)
    # Return the closest point
    return B[min_index]

## TODO: Why do we need to check the axis?
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

def check_axis(rotmat):
    x_axis = rotmat[:,2]
    # y_axis = rotmat[:,1]
    # z_axis = rotmat[:,2]
    angle_x = abs(np.dot(x_axis, np.array([1, 0, 0])) ) # x 轴方向
    angle_y = abs(np.dot(x_axis, np.array([0, 1, 0])))  # y 轴方向
    angle_z = abs(np.dot(x_axis, np.array([0, 0, 1])))  # z 轴方向

    # 找出夹角最大的轴作为机器人的主要朝向
    main_orientation = max(angle_x, angle_y, angle_z)

    if main_orientation == angle_x:
        return 0
    elif main_orientation == angle_y:
        return 1
    else:
        return 2
    
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])
  
def get_distance(pose1, pose2):
    return (pose1[0]- pose2[0]) ** 2 + (pose1[1]- pose2[1]) ** 2 + (pose1[2]- pose2[2]) ** 2

def rotate_quaternion(quaternion, axis, angle):
    # 将四元数和轴转换为NumPy数组
    quaternion = np.array(quaternion)
    axis = np.array(axis)

    # 将轴向量归一化为单位向量
    axis = axis / np.linalg.norm(axis)

    # 计算旋转的一半角度
    half_angle = angle / 2

    # 创建表示旋转的四元数
    rotation_quaternion = np.array([np.cos(half_angle),
                                    axis[0] * np.sin(half_angle),
                                    axis[1] * np.sin(half_angle),
                                    axis[2] * np.sin(half_angle)])

    # 进行四元数乘法，得到旋转后的四元数
    rotated_quaternion = quaternion_multiply(rotation_quaternion, quaternion)

    return rotated_quaternion

def place_motion(target_quat,world_point):
    # print(quaternion_to_rotation_matrix(target_quat))
    gripper_axis = check_axis(quaternion_to_rotation_matrix(target_quat))
    print(gripper_axis)
    # exit()
    # if gripper_axis == 2: #negative z-axis: downward
    # T_ee_world = fa.get_pose()
    # random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
    #     from_frame='franka_tool', to_frame='world')
    # # # random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.02,0.01,-0.1 ]),
    # # #     from_frame='franka_tool', to_frame='world')
    # fa.goto_pose(random_position) 
    # exit()
    if gripper_axis == 0:
        random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.18,0.05,-0.15 ]),
        from_frame='franka_tool', to_frame='world')
        fa.goto_pose(random_position) 
        # exit()
    # exit()
        T_ee_world = fa.get_pose()
        # print(T_ee_world.translation)
        T_ee_world.translation -= [-0.18, 0, -0.0]
        fa.goto_pose(T_ee_world)
        # exit()
        T_ee_world = fa.get_pose()
    # print(T_ee_world.translation)
        fa.open_gripper()
        
def quaternion_cosine_similarity(q1, q2):
    # Compute dot product of quaternions
    dot_product = np.dot(q1, q2)

    # Compute magnitudes (norms) of quaternions
    norm_q1 = np.linalg.norm(q1)
    norm_q2 = np.linalg.norm(q2)

    # Compute cosine similarity
    cosine_similarity = dot_product / (norm_q1 * norm_q2)
    return cosine_similarity

def move_motion(target_quat,world_point):
  '''
  pick along the z axis
  '''
  #go to the up direction of contact point
  random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.02,0.01,-0.00]),
      from_frame='franka_tool', to_frame='world')

  fa.goto_pose(random_position) 
  T_ee_world = fa.get_pose()
  T_ee_world.translation += [0, 0, 0.2]
  fa.goto_pose(T_ee_world)
  
def rotate_motion(target_quat,world_point):
  '''
  pick along the z axis
  '''
#   fa.open_gripper()
#   T_ee_world = fa.get_pose()
# #   random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
# #       from_frame='franka_tool', to_frame='world')
  gripper_axis = check_axis(quaternion_to_rotation_matrix(T_ee_world.quaternion))
# #   random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
# #       from_frame='franka_tool', to_frame='world')
# #   fa.goto_pose(random_position) 
#   best_pose = RigidTransform(rotation=[ 0.7162478,  -0.0705974 ,  0.69274537,  0.04587066], translation=[0.54341077, 0.07044586, 0.59715089],
#         from_frame='franka_tool', to_frame='world')
#   fa.goto_pose(best_pose)
#   fa.close_gripper()
#   joints = fa.get_joints()
#   joints[6] += np.deg2rad(60)
#   fa.goto_joints(joints)
#   fa.open_gripper()
#   print(gripper_axis)
# #   exit()
  if gripper_axis ==0: 
#     random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.05,0.05,0.0]),
#         from_frame='franka_tool', to_frame='world')

#     fa.goto_pose(random_position) 
    T_ee_world = fa.get_pose()
#     # exit()
#     #   print(target_quat,T_ee_world.quaternion)
#     # exit()
#     #move downward
#     random_position = RigidTransform(rotation=target_quat, translation=np.array(T_ee_world.translation-[-0.05,0,0]),
#         from_frame='franka_tool', to_frame='world')

#     fa.goto_pose(random_position) 
#     T_ee_world = fa.get_pose()
#     print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    #   exit()
    fa.close_gripper()
    # exit()
    
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0, 0, -0.03]
    fa.goto_pose(T_ee_world)
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0, 0, -0.04]
    fa.goto_pose(T_ee_world)
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0, 0, -0.04]
    fa.goto_pose(T_ee_world)
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0, 0, -0.03]
    fa.goto_pose(T_ee_world)
    exit()
  
def pick_motion(target_quat,world_point):
  '''
  pick along the z axis
  '''
  #go to the up direction of contact point
  fa.open_gripper()
  T_ee_world = fa.get_pose()
  gripper_axis = check_axis(quaternion_to_rotation_matrix(T_ee_world.quaternion))
#   random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
#       from_frame='franka_tool', to_frame='world')
  if gripper_axis ==2: #negative z-axis: downward
    random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.03,0.05,-0.03]),
        from_frame='franka_tool', to_frame='world')

    fa.goto_pose(random_position) 
    T_ee_world = fa.get_pose()
    #   exit()
    #   print(target_quat,T_ee_world.quaternion)
    # exit()
    #move downward
    random_position = RigidTransform(rotation=target_quat, translation=np.array(T_ee_world.translation-[0,0,0.13]),
        from_frame='franka_tool', to_frame='world')

    fa.goto_pose(random_position) 
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    #   exit()
    fa.close_gripper()
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0, 0, 0.1]
    fa.goto_pose(T_ee_world)
  # exit()
  # fa.open_gripper()

def pick_motion2(target_quat,world_point):
  '''
  pick along the z axis
  '''
  #go to the up direction of contact point
  fa.open_gripper()
  T_ee_world = fa.get_pose()
  
  print("T_ee_world: ", T_ee_world)
  gripper_axis = check_axis(quaternion_to_rotation_matrix(T_ee_world.quaternion))
  print('gripper_axis:', gripper_axis)
  if gripper_axis ==2: #negative z-axis: downward
    random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]),
        from_frame='franka_tool', to_frame='world')

    fa.goto_pose(random_position)
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    fa.close_gripper()

def push_motion(target_quat,world_point,NUM_CONTROL=4,MOVE_DELTA=0.03):
    print('-----------------establish initial contact')
    fa.close_gripper()
    T_ee_world = fa.get_pose()
    # random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
    #     from_frame='franka_tool', to_frame='world')
    # fa.goto_pose(random_position) 
    gripper_axis = check_axis(quaternion_to_rotation_matrix(T_ee_world.quaternion))
    # print(gripper_axis)
    # exit()
    
    if gripper_axis ==  0: #positive x-axis: forward
        random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.14,0.1,0.01]),
        from_frame='franka_tool', to_frame='world')
        fa.goto_pose(random_position) 
        # exit()
        T_ee_world = fa.get_pose()
        random_position = RigidTransform(rotation=target_quat, translation=np.array(T_ee_world.translation+ [0.05,0,0]),
            from_frame='franka_tool', to_frame='world')
        fa.goto_pose(random_position) 
        
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    print('-----------------start gradually pulling and adjusting pose')
    for i in range(NUM_CONTROL):
        #calculate the next moving position
        init_pose = fa.get_pose()
        init_quat = init_pose.quaternion
        if gripper_axis == 2:
            init_dir = np.array([0, 1, 0]) # grasp edge: gripper direction, z-axis
        elif gripper_axis == 0:
            init_dir = np.array([0, 0, 1]) # grasp handle: gtipper forward direction x-axis
        rotation_quaternion = Quaternion(init_quat[0], init_quat[1], init_quat[2], init_quat[3])
        init_dir = Quaternion(0, init_dir[0], init_dir[1], init_dir[2])
        result_quaternion = rotation_quaternion * init_dir * rotation_quaternion.conjugate
        move_dir = [result_quaternion.x, result_quaternion.y, result_quaternion.z]
        move_pos = init_pose.translation + [MOVE_DELTA * val for val in move_dir]
        # print(move_pos,init_pose.translation)
        # exit()
        next_pose = RigidTransform(rotation=init_quat, translation=move_pos,
        from_frame='franka_tool', to_frame='world')
        fa.goto_pose(next_pose)
        
        
        #calculate the adjusted quaternion
        current_pose = fa.get_pose()  
        delta_vec  = current_pose.translation - init_pose.translation
        delta_s2 = get_distance(current_pose.translation,init_pose.translation)
        delta_pos = [MOVE_DELTA * val for val in move_dir]
        delta_l = np.dot(delta_vec, delta_pos) / np.linalg.norm(delta_pos)
        delta_x = np.sqrt(delta_s2 - delta_l ** 2)
        delta_theta = np.arcsin(delta_x * delta_l / delta_s2)
        axis = np.cross(delta_vec, delta_pos)
        final_quat = rotate_quaternion(current_pose.quaternion, axis, delta_theta) 
        
        next_pose = RigidTransform(rotation=final_quat, translation=current_pose.translation,
        from_frame='franka_tool', to_frame='world')
        fa.goto_pose(next_pose)
        
        
    
  
def pull_motion(target_quat,world_point,NUM_CONTROL=4,MOVE_DELTA=0.03):
  print('-----------------establish initial contact')
  fa.open_gripper()
  T_ee_world = fa.get_pose()
  gripper_axis = check_axis(quaternion_to_rotation_matrix(target_quat))
#   gripper_axis = 0
#   random_position = RigidTransform(rotation=target_quat, translation=T_ee_world.translation,
#         from_frame='franka_tool', to_frame='world')
#   fa.goto_pose(random_position) 
#   exit()
  if gripper_axis ==  0: #positive x-axis: forward
    random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.27,0.05,0]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose(random_position) 
    T_ee_world = fa.get_pose()
    random_position = RigidTransform(rotation=target_quat, translation=np.array(T_ee_world.translation+ [0.08,0,0]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose(random_position) 
    
  elif gripper_axis == 2: #negative z-axis: downward
    # exit()
    random_position = RigidTransform(rotation=target_quat, translation=np.array(world_point[:3]-[0.03,0.05,-0.05]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose(random_position) 
    # exit()
    T_ee_world = fa.get_pose()
    random_position = RigidTransform(rotation=target_quat, translation=np.array(T_ee_world.translation-[0.0,0.01,0.07]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose(random_position) 
#   exit()
  T_ee_world = fa.get_pose()
  print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
  fa.close_gripper()
#   exit()

  print('-----------------start gradually pulling and adjusting pose')
  for i in range(NUM_CONTROL):
    #calculate the next moving position
    init_pose = fa.get_pose()
    init_quat = init_pose.quaternion
    
    if gripper_axis == 2:
        init_dir = np.array([0, 1, 0]) # grasp edge: gripper direction, z-axis
    elif gripper_axis == 0:
        init_dir = np.array([0, 0, 1]) # grasp handle: gtipper forward direction x-axis
    
    rotation_quaternion = Quaternion(init_quat[0], init_quat[1], init_quat[2], init_quat[3])
    init_dir = Quaternion(0, init_dir[0], init_dir[1], init_dir[2])
    result_quaternion = rotation_quaternion * init_dir * rotation_quaternion.conjugate
    move_dir = [result_quaternion.x, result_quaternion.y, result_quaternion.z]
    move_pos = init_pose.translation + [MOVE_DELTA * val for val in move_dir]
    # print(move_pos,init_pose.translation)
    # exit()
    next_pose = RigidTransform(rotation=init_quat, translation=move_pos,
    from_frame='franka_tool', to_frame='world')
    fa.goto_pose(next_pose)
    
    
    #calculate the adjusted quaternion
    current_pose = fa.get_pose()  
    delta_vec  = current_pose.translation - init_pose.translation
    delta_s2 = get_distance(current_pose.translation,init_pose.translation)
    delta_pos = [MOVE_DELTA * val for val in move_dir]
    delta_l = np.dot(delta_vec, delta_pos) / np.linalg.norm(delta_pos)
    delta_x = np.sqrt(delta_s2 - delta_l ** 2)
    delta_theta = np.arcsin(delta_x * delta_l / delta_s2)
    axis = np.cross(delta_vec, delta_pos)
    final_quat = rotate_quaternion(current_pose.quaternion, axis, -delta_theta)
    print(final_quat,current_pose.quaternion)
    # exit()
    next_pose = RigidTransform(rotation=final_quat, translation=current_pose.translation,
    from_frame='franka_tool', to_frame='world')
    fa.goto_pose(next_pose)
    
    

  
def rotmat_to_quaternion(rotmat):
    # Compute the trace of the rotation matrix
    trace = np.trace(rotmat)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotmat[2, 1] - rotmat[1, 2]) * s
        qy = (rotmat[0, 2] - rotmat[2, 0]) * s
        qz = (rotmat[1, 0] - rotmat[0, 1]) * s
    elif rotmat[0, 0] > rotmat[1, 1] and rotmat[0, 0] > rotmat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2])
        qw = (rotmat[2, 1] - rotmat[1, 2]) / s
        qx = 0.25 * s
        qy = (rotmat[0, 1] + rotmat[1, 0]) / s
        qz = (rotmat[0, 2] + rotmat[2, 0]) / s
    elif rotmat[1, 1] > rotmat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2])
        qw = (rotmat[0, 2] - rotmat[2, 0]) / s
        qx = (rotmat[0, 1] + rotmat[1, 0]) / s
        qy = 0.25 * s
        qz = (rotmat[1, 2] + rotmat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1])
        qw = (rotmat[1, 0] - rotmat[0, 1]) / s
        qx = (rotmat[0, 2] + rotmat[2, 0]) / s
        qy = (rotmat[1, 2] + rotmat[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])
  
def result2rotmat(result,extrinsic_matrix):
  d_x, d_y, d_z = result.split('[')[1].split(']')[0].split(', ')
  gripper_up_direction_world = np.array([int(d_x)*0.02, int(d_y)*0.02, int(d_z)*0.02])

  fd_x, fd_y, fd_z = result.split('[')[2].split(']')[0].split(', ')
  gripper_left_direction_world = np.array([int(fd_x)*0.02, int(fd_y)*0.02, int(fd_z)*0.02])
  T_ee_world = fa.get_pose()
  print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
  init_quat = T_ee_world.quaternion
  init_rot = quaternion_to_rotation_matrix(init_quat)
  init_left = init_rot[:3, 1]
  cos_simi = np.dot(init_left,gripper_left_direction_world)
  if cos_simi<0:
    print('------init and pred is not in the same hemispherem, flip the left direction------')
    gripper_left_direction_world *=-1
  # gripper_up_direction_world = extrinsic_matrix[:3,:3] @ gripper_up_direction_world
  gripper_up_direction_world /= np.linalg.norm(gripper_up_direction_world)
  # gripper_left_direction_world = extrinsic_matrix[:3,:3] @ gripper_left_direction_world
  gripper_left_direction_world /= np.linalg.norm(gripper_left_direction_world)

  gripper_forward_direction_world = np.cross(gripper_left_direction_world, gripper_up_direction_world)
  gripper_forward_direction_world /= np.linalg.norm(gripper_forward_direction_world)
  gripper_left_direction_world = np.cross(gripper_up_direction_world, gripper_forward_direction_world)
  gripper_left_direction_world /= np.linalg.norm(gripper_left_direction_world)
  contact_rotmat = np.eye(4).astype(np.float32)
  contact_rotmat[:3, 0] = gripper_forward_direction_world
  contact_rotmat[:3, 1] = gripper_left_direction_world
  contact_rotmat[:3, 2] = gripper_up_direction_world
#   contact_rotmat[:3, 0] = gripper_up_direction_world
#   contact_rotmat[:3, 1] = gripper_left_direction_world
#   contact_rotmat[:3, 2] = gripper_forward_direction_world
  x, y = result.split('(')[1].split(')')[0].split(', ')
  x = int(x)
  y = int(y)
  return x,y,contact_rotmat
  
def depth_image_to_point_cloud(depth_image,color_intrinsics):
    rows, cols = depth_image.shape
    points = []
    rgbd = np.zeros((rows, cols, 3))
    # permutation = np.array(([0,-1,0],[-1,0,0],[0,0,-1]))
    for y in range(rows):
        for x in range(cols):
            depth = depth_image[y, x]
            if depth > 0:
                z_c = depth *0.001  # Convert depth from mm to meters
                x_c = (x -  color_intrinsics.ppx) * z_c / color_intrinsics.fx
                y_c = (y - color_intrinsics.ppy) * z_c / color_intrinsics.fy
                points.append(np.array([x_c, y_c, z_c]))
    xs,ys = np.where(depth_image>0)
    rgbd[xs,ys] = points
    return points, rgbd

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    q = np.array(quaternion) / np.linalg.norm(quaternion)

    w, x, y, z = q[0], q[1], q[2], q[3]

    # Compute rotation matrix elements
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                  [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                  [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])

    return R

def convert_to_image_points(camera_coords, intrinsic_matrix):
    # Extract relevant parts from the intrinsic matrix
    K = intrinsic_matrix[:3, :3]
    cx, cy = intrinsic_matrix[:2, 2]

    # Extract camera coordinates
    X, Y, Z = camera_coords.flatten()

    # Apply the pinhole camera model equations
    coords_homogeneous = np.array([[X], [Y], [Z]])
    projection_matrix = K @ coords_homogeneous
    u = projection_matrix[0] / projection_matrix[2] + cx
    v = projection_matrix[1] / projection_matrix[2] + cy

    return u, v
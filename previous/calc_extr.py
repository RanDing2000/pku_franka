import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial.transform import Rotation as R

# Load color and depth data (assuming these are already loaded into color_data and depth_img)
f = np.load('/home/hyperpanda/Haoran/scenes/demo_pattern/single_scene/raw.npz', allow_pickle=True)
color_img = f['color']
depth_img = f['depth']

# Camera intrinsic parameters
focal_length_x = 913.576
focal_length_y = 912.938
principal_point_x = 628.32
principal_point_y = 360.564

# ROI coordinates (replace these with your actual coordinates)
x, y, w, h = 657, 280, 113, 104

# Extract the ROI from the depth image
roi_depth = depth_img[y:y+h, x:x+w]

# Generate meshgrid for the pixel coordinates
xx, yy = np.meshgrid(range(x, x + w), range(y, y + h))

# Convert pixel coordinates and depth to 3D coordinates
zz = roi_depth.flatten()
xx = (xx.flatten() - principal_point_x) * zz / focal_length_x
yy = (yy.flatten() - principal_point_y) * zz / focal_length_y

# Fit a plane to the 3D points using RANSAC
X = np.column_stack([xx, yy])
y = zz
ransac = make_pipeline(PolynomialFeatures(1), RANSACRegressor(random_state=0))
ransac.fit(X, y)

# Extract the plane coefficients
coef = ransac.named_steps['ransacregressor'].estimator_.coef_
intercept = ransac.named_steps['ransacregressor'].estimator_.intercept_
normal_vector = np.array([coef[1], coef[2], -1])
normal_vector = normal_vector / np.linalg.norm(normal_vector)

# Calculate the rotation matrix
camera_forward = np.array([0, 0, 1])
rotation_axis = np.cross(camera_forward, normal_vector)
rotation_angle = np.arccos(np.dot(camera_forward, normal_vector) / np.linalg.norm(normal_vector))
rotation = R.from_rotvec(rotation_axis * rotation_angle)
rotation_matrix = rotation.as_matrix()

# Calculate the translation vector
z_0 = -intercept / normal_vector[2]
translation_vector = np.array([0, 0, z_0])

print("Rotation Matrix:")
print(rotation_matrix)
print("\nTranslation Vector:")
print(translation_vector)

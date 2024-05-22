import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from numpy.linalg import svd, det
import argparse
import json

def load_data(path):
    data = np.load(path)
    return data['color'], data['depth']

def display_image_for_selection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Set the title of the figure with the warning message
    fig.suptitle('Click to select corners in the order of: bottom left (origin), bottom right, upper left, upper right!', fontsize=12, color='red')

    coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            ax.plot(x, y, 'ro')
            fig.canvas.draw()
            print(f"Point selected: ({x}, {y})")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return coords

def refine_corners(image, coords, radius=5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    refined_coords = []
    for x, y in coords:
        # Check boundaries and adjust if necessary
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(image.shape[1], x + radius + 1)
        y_max = min(image.shape[0], y + radius + 1)
        
        # Adjust the search box size based on the boundaries
        search_box_size = (y_max - y_min, x_max - x_min)
        
        local_max = np.unravel_index(np.argmax(dst[y_min:y_max, x_min:x_max]), search_box_size)
        refined_coords.append((x_min + local_max[1], y_min + local_max[0]))

    return refined_coords

def calc_corners(color_image):
    # Display image for manual selection
    initial_coords = display_image_for_selection(color_image)
    # Refine selected coordinates using Harris Corner Detection
    refined_corners = refine_corners(color_image, initial_coords)
    return refined_corners

# Define the function to convert 2D image coordinates to 3D points correctly within this context
def image_to_3d(depth, points, focal_length, principal_point):
    points_3d = []
    for (x, y) in points:
        Z = depth[y, x] / 1000 # Depth value at (x, y)
        if Z > 0:  # Valid depth value
            X = (x - principal_point[0]) * Z / focal_length[0]
            Y = (y - principal_point[1]) * Z / focal_length[1]
            points_3d.append((X, Y, Z))
        else:  # Invalid depth value
            points_3d.append((None, None, None))
    return np.array(points_3d)

# Project the 3D points back onto the 2D image plane
def project_to_image(points_3d, focal_length, principal_point):
    projected_2d = []
    for x, y, z, _ in points_3d:
        u = (focal_length[0] * x / z) + principal_point[0]
        v = (focal_length[1] * y / z) + principal_point[1]
        projected_2d.append((u, v))
    return projected_2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and save a transformation matrix between plane coordinates and camera coordinates.')
    parser.add_argument("--save_dir_root",type=str,default="/home/hyperpanda/Haoran")
    args = parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    # Path to your .npz file
    raw_npz_path = os.path.join(save_dir, 'raw.npz')
    color_image, depth_image = load_data(raw_npz_path)
    envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    refined_corners = calc_corners(color_image)
    print("Refined coordinates:", refined_corners)

    intrinsics = np.load(os.path.join(save_dir, "intrinsics.npy"), allow_pickle=True)
    focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
    principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

    # Convert the refined corner coordinates to 3D points again
    camera_corners_3d = image_to_3d(depth_image, refined_corners, focal_length, principal_point)

    # Assuming the whiteboard's coordinate system with the first point as origin
    # and creating a coordinate system based on the provided points
    # We create a local coordinate system for the whiteboard assuming the first corner as the origin (0, 0, 0)
    # The x-axis is defined towards the second point, and the y-axis towards the fourth, with appropriate scaling
    whiteboard_points_3d = np.array([
        [0, 0, 0.05],  # Origin
        [0.307, 0, 0.05],  # X-axis direction
        [0, 0.307, 0.05],  # Y-axis direction
        [0.307, 0.307, 0.05],  # XY plane
        [0.1, 0.1, 0.05],
        [0.1, 0.2, 0.05],
        [0.2, 0.1, 0.05],
        [0.2, 0.2, 0.05]
    ])

    # Compute centroids of both sets
    centroid_camera = np.mean(camera_corners_3d, axis=0)
    centroid_whiteboard = np.mean(whiteboard_points_3d, axis=0)

    # Translate points by subtracting centroids
    camera_points_centered = camera_corners_3d - centroid_camera
    whiteboard_points_centered = whiteboard_points_3d - centroid_whiteboard

    # print these all
    print("camera_corners_3d: \n", camera_corners_3d)
    print("centroid_camera: \n", centroid_camera)
    print("whiteboard_points_3d: \n", whiteboard_points_3d)
    print("centroid_whiteboard: \n", centroid_whiteboard)

    # Compute the Cross-Covariance Matrix H
    H = np.dot(camera_points_centered.T, whiteboard_points_centered)
    U, _, Vt = svd(H)

    R = np.dot(Vt.T, U.T)
    # Correct the Rotation Matrix in case of reflection
    if det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the Translation Vector t
    t = centroid_whiteboard - np.dot(R, centroid_camera)

    # Construct the Transformation Matrix T
    T = np.eye(4)  # Initialize a 4x4 identity matrix
    T[:3, :3] = R  # Set the upper left 3x3 to the rotation matrix
    T[:3, 3] = t  # Set the last column to the translation vector

    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", t)
    print("Transformation Matrix (T):\n", T)
    print("=========== start testing ============")                                                        

    # origin = np.array([[0, 0, 0.02, 1]]).T
    # cam_origin_3d = T @ origin
    # print("done")

    # Assuming R and t have been defined earlier as part of the SE(3) transformation calculation
                                                                                                                                                           
    R_inv = R.T  # Transpose of R for the inverse rotation
    t_inv = -np.dot(R_inv, t)  # Inverse translation

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)  # Initialize a 4x4 identity matrix
    T_inv[:3, :3] = R_inv  # Upper left 3x3 submatrix is the inverse rotation
    T_inv[:3, 3] = t_inv  # Last column is the inverse translation

    # Define the whiteboard's points in its coordinate system
    # Assuming these points correspond to the corners in the whiteboard's local coordinate system
    whiteboard_points_homogeneous = np.hstack((whiteboard_points_3d, np.ones((whiteboard_points_3d.shape[0], 1))))

    # Transform the whiteboard's points back to the camera's coordinate system
    camera_points_transformed = np.dot(T_inv, whiteboard_points_homogeneous.T).T

    # Project the transformed 3D points back onto the 2D image plane
    projected_points = project_to_image(camera_points_transformed, focal_length, principal_point)
    print("Projected 2D Points on Image:", projected_points)

    # mark these projected points on color image
    fig, ax = plt.subplots()
    ax.imshow(color_image)
    for u, v in projected_points:
        ax.plot(u, v, 'ro')
    plt.savefig("projected_points.png")
    print("=========== end testing ============")

    # Define the output filename using the provided scene_id
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save transformation matrix to .npy file with scene_id in the filename
    filename = os.path.join(save_dir, 'cam2plane_transformation.npy')
    np.save(filename, T)  # Assuming T is your transformation matrix

    print(f"Transformation matrix saved as '{filename}'")

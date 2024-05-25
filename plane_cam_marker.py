import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import json
from utils_scene import inverse_extrinsics


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
            ax.plot(x, y, 'ro', markersize=2)
            fig.canvas.draw()
            print(f"Point selected: ({x}, {y})")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return coords

def refine_corners(image, coords, radius=11):
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


def marker_estimatePose(image, cameraMatrix, distCoeffs, markerLength):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(image, dictionary)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        markerCorners, markerLength, cameraMatrix, distCoeffs)
    cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

    if rvecs is None or len(rvecs) == 0:
        return False, rvecs, tvecs, image

    return True, rvecs, tvecs, image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and save a transformation matrix between plane coordinates and camera coordinates.')
    parser.add_argument("--save_dir_root", type=str, default="")
    args = parser.parse_args()
    with open(os.path.join(args.save_dir_root, "config.json"), "r") as config_file:
        config = json.load(config_file)
    scene_id = config["scene_id"]
    save_dir = os.path.join(args.save_dir_root, 'scenes', scene_id)
    # Path to your .npz file
    raw_npz_path = os.path.join(save_dir, 'raw.npz')
    # color_image, depth_image = load_data(raw_npz_path)
    color_image = load_data(raw_npz_path)[0]
    # envpath = "/home/hyperpanda/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    # refined_corners = np.asarray(calc_corners(color_image), dtype=np.float32)
    # print("Refined coordinates:", refined_corners)
    # # save the color image with the refined coordinates marked
    # fig, ax = plt.subplots()
    # ax.imshow(color_image)
    # for x, y in refined_corners:
    #     # smaller blue circle
    #     ax.plot(x, y, 'bo', markersize=1)
    # plt.savefig("refined_corners.png")
    
    intrinsics = np.load(os.path.join(save_dir, "intrinsics.npy"), allow_pickle=True)
    focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
    principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

    # Convert the refined corner coordinates to 3D points again
    # camera_corners_3d = image_to_3d(depth_image, refined_corners, focal_length, principal_point)

    # Assuming the whiteboard's coordinate system with the first point as origin
    # and creating a coordinate system based on the provided points
    # We create a local coordinate system for the whiteboard assuming the first corner as the origin (0, 0, 0)
    # The x-axis is defined towards the second point, and the y-axis towards the fourth, with appropriate scaling
    whiteboard_points_3d = np.array([
        [0, 0, 0.05],  # Origin
        [0.307, 0, 0.05],  # X-axis direction
        [0, 0.307, 0.05],  # Y-axis direction
        [0.307, 0.307, 0.05],  # XY plane
    ], dtype=np.float32)
    
    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve for the extrinsic parameters
    # ret, rvec, t = cv2.solvePnP(
    #     whiteboard_points_3d, refined_corners, intrinsics, dist_coeffs)

    marker_length = 0.05
    # ret, rvecs, tvecs, image = multi_marker_estimatePose3(image, cameraMatrix, distCoeffs, markerLength)
    ret, rvec, t, image = marker_estimatePose(
        color_image, intrinsics, dist_coeffs, marker_length)
    t = t.reshape(3, 1)
    t = t
    
    if ret:
        cv2.drawFrameAxes(image, intrinsics, dist_coeffs, rvec, t, 0.05)
        cv2.imshow("Image with Pose", color_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    T = np.eye(4)  # Initialize a 4x4 identity matrix
    T[:3, :3] = R  # Set the upper left 3x3 to the rotation matrix
    T[:3, 3] = t.flatten()  # Set the last column to the translation vector
    
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", t)
    print("Transformation Matrix (T):\n", T)
    print("=========== start testing ============")                                                        

    origin = np.array([[0, 0, 0.05, 1]]).T
    cam_origin_3d = T @ origin
    print(f"cordinate of origin in camera frame: {cam_origin_3d}")

    # Assuming R and t have been defined earlier as part of the SE(3) transformation calculation
    R_inv = R.T  # Transpose of R for the inverse rotation
    t_inv = -np.dot(R_inv, t)  # Inverse translation

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)  # Initialize a 4x4 identity matrix
    T_inv[:3, :3] = R_inv  # Upper left 3x3 submatrix is the inverse rotation
    T_inv[:3, 3] = t_inv.flatten() - np.array([0.25, 0.25, 0]).reshape(3, 1)  # Last column is the inverse translation
    
    T = inverse_extrinsics(T_inv)
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(color_image, intrinsics, dist_coeffs, rvec, t, 0.05)
    cv2.imshow("Image with Pose", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Define the whiteboard's points in its coordinate system
    # Assuming these points correspond to the corners in the whiteboard's local coordinate system
    whiteboard_points_homogeneous = np.hstack((whiteboard_points_3d, np.ones((whiteboard_points_3d.shape[0], 1))))

    # Transform the whiteboard's points back to the camera's coordinate system
    camera_points_transformed = np.dot(T, whiteboard_points_homogeneous.T).T

    # Project the transformed 3D points back onto the 2D image plane
    projected_points = project_to_image(camera_points_transformed, focal_length, principal_point)
    print("Projected 2D Points on Image:", projected_points)

    # mark these projected points on color image
    fig, ax = plt.subplots()
    ax.imshow(color_image)
    for u, v in projected_points:
        ax.plot(u, v, 'go', markersize=1)
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

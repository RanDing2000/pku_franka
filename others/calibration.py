import cv2
import numpy as np

# Load the image
dir = "/home/hyperpanda/Haoran/scenes/demo_pattern/single_scene"
image = np.load(f"{dir}/raw.npz", allow_pickle=True)['color']

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris corner detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Result is dilated for marking the corners
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, marking the corners in red
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# Save the image with the Harris corners
output_filename = f'{dir}/harris_corners.png'
cv2.imwrite(output_filename, image)

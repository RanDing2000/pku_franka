import cv2
import numpy as np

f = np.load('/home/hyperpanda/Haoran/scenes/demo_pattern/single_scene/raw.npz', allow_pickle=True)
color_img = f['color']
depth_img = f['depth']

# Read image
# image = cv2.imread("image.png") 
  
# Select ROI 
r = cv2.selectROI("select the area", color_img) 
# (657, 280, 113, 104)

# Crop image 
cropped_image = color_img[int(r[1]):int(r[1]+r[3]),  
                      int(r[0]):int(r[0]+r[2])] 
  
# Display cropped image 
cv2.imshow("Cropped image", cropped_image) 
cv2.waitKey(0)
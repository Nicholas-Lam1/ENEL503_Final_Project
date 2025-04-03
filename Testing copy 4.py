import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

# Directory containing the images
image_path = "./Dataset/test/b526b661a33ff481231d1342aff2a266_jpg.rf.dd4da31fe7f3f0eac58576f8f2c56f61.jpg"
image = cv2.imread(image_path)

cv2.imshow("Original", image)
cv2.imshow("Blue", image[:,:,0])
cv2.imshow("Green", image[:,:,1])
cv2.imshow("Red", image[:,:,2])

cv2.waitKey(0)
cv2.destroyAllWindows()

blue_channel = image[:,:,0]

min_val = np.min(blue_channel) 
max_val = np.max(blue_channel) 

stretched_blue = ((blue_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

cv2.imshow("Contrast Stretched Blue", stretched_blue)

green_channel = image[:,:,1]

min_val = np.min(green_channel) 
max_val = np.max(green_channel) 

stretched_green = ((green_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

cv2.imshow("Contrast Stretched Green", stretched_green)

red_channel = image[:,:,2]

min_val = np.min(red_channel) 
max_val = np.max(red_channel) 

stretched_red = ((red_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

cv2.imshow("Contrast Stretched Red", stretched_red)

stretched_rgb = cv2.merge([stretched_blue, stretched_green, stretched_red])

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([stretched_rgb],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Original", image)
cv2.imshow("Contrast Stretched", stretched_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

blur_small = cv2.GaussianBlur(stretched_rgb, (5, 5), 0) 
blur_large = cv2.GaussianBlur(stretched_rgb, (5, 5), 1)  

dog_image = cv2.subtract(blur_small, blur_large)
dog_image = cv2.normalize(dog_image, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Difference of Gaussian", dog_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

edges = cv2.subtract(stretched_rgb, blur_small)
boosted_image = cv2.add(stretched_rgb, edges)

cv2.imshow("Contrast Stretched", stretched_rgb)
cv2.imshow("Boosted Edges", boosted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

adaptive_thresh_5 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
adaptive_thresh_7 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
adaptive_thresh_11 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_thresh_13 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
adaptive_thresh_15 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
adaptive_thresh_17 = cv2.adaptiveThreshold(stretched_green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

cv2.imshow("Adaptive Thresh - 5", adaptive_thresh_5)
cv2.imshow("Adaptive Thresh - 7", adaptive_thresh_7)
cv2.imshow("Adaptive Thresh - 11", adaptive_thresh_11)
cv2.imshow("Adaptive Thresh - 13", adaptive_thresh_13)
cv2.imshow("Adaptive Thresh - 15", adaptive_thresh_15)
cv2.imshow("Adaptive Thresh - 17", adaptive_thresh_17)

cv2.waitKey(0)
cv2.destroyAllWindows()

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True  # Filter blobs by area size
params.minArea = 100  # Minimum area of a blob
params.maxArea = 2000  # Maximum area of a blob

params.filterByCircularity = True 
params.minCircularity = 0.1  

params.filterByConvexity = True  
params.minConvexity = 0.1  
params.maxConvexity = 0.8  

params.filterByInertia = True  
params.minInertiaRatio = 0.1  
params.maxInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create()
detector.setParams(params)

keypoints = detector.detect(stretched_blue)

result_image = cv2.drawKeypoints(stretched_blue, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blobs", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# stretched_grey = cv2.cvtColor(stretched_rgb, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Original", image_grey)
# cv2.imshow("Contrast Stretched", stretched_grey)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([stretched_rgb],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()


# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([image],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

#unsharpening
#contrast stretched
#adaptive thresholding
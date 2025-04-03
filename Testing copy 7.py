import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import shutil


# Directory containing the images
image_directory = "./Dataset/test"  # Replace with the path to your directory
output_directory = "./Output"  # Replace with your output directory

if os.path.exists(output_directory):
    shutil.rmtree(output_directory)  # Deletes the entire directory and its contents

os.makedirs(output_directory)  # Recreate the empty directory

# Step 1: Process all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp")):  # Supported image formats
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not open or find the image: {filename}")
            continue

        # cv2.imshow("Original", image)
        # cv2.imshow("Blue", image[:,:,0])
        # cv2.imshow("Green", image[:,:,1])
        # cv2.imshow("Red", image[:,:,2])

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        blue_channel = image[:,:,0]

        min_val = np.min(blue_channel) 
        max_val = np.max(blue_channel) 

        stretched_blue = ((blue_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # cv2.imshow("Contrast Stretched Blue", stretched_blue)

        green_channel = image[:,:,1]

        min_val = np.min(green_channel) 
        max_val = np.max(green_channel) 

        stretched_green = ((green_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # cv2.imshow("Contrast Stretched Green", stretched_green)

        red_channel = image[:,:,2]

        min_val = np.min(red_channel) 
        max_val = np.max(red_channel) 

        stretched_red = ((red_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # cv2.imshow("Contrast Stretched Red", stretched_red)

        stretched_rgb = cv2.merge([stretched_blue, stretched_green, stretched_red])

        # cv2.imshow("Original", image)
        # cv2.imshow("Contrast Stretched", stretched_rgb)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        lower_bound = np.array([0, 0, 100])  
        upper_bound = np.array([210, 160, 255])  

        mask = cv2.inRange(stretched_rgb, lower_bound, upper_bound)

        lower_bound = np.array([180, 230, 230])  
        upper_bound = np.array([255, 255, 255])  

        mask2 = cv2.inRange(stretched_rgb, lower_bound, upper_bound)

        result = cv2.bitwise_and(stretched_rgb, stretched_rgb, mask=mask)
        result2 = cv2.bitwise_and(stretched_rgb, stretched_rgb, mask=mask2)

        black_background = np.zeros_like(stretched_rgb)  
        summed_mask = cv2.add(result, result2)
        final_mask = cv2.add(black_background, summed_mask)

        kernel = np.ones((1, 3) , np.uint8)
        vert_line_remove = cv2.erode(final_mask, kernel, iterations=1)

        kernel = np.ones((1, 5) , np.uint8)
        vert_join1 = cv2.dilate(vert_line_remove, kernel, iterations=1)

        kernel = np.ones((1, 5) , np.uint8)
        vert_erode1 = cv2.erode(vert_join1, kernel, iterations=1)

        kernel = np.ones((1, 5) , np.uint8)
        vert_join2 = cv2.dilate(vert_join1, kernel, iterations=1)

        kernel = np.ones((1, 5) , np.uint8)
        vert_erode2 = cv2.erode(vert_join2, kernel, iterations=1)
    
        kernel = np.ones((5, 5) , np.uint8)
        close = cv2.morphologyEx(vert_erode2, cv2.MORPH_CLOSE, kernel)

        gray = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)

        # Threshold the image (binary inverse to detect dark regions)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)        

        # Find contours of the blobs
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        for contour in contours:
            # Optionally filter by blob size
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust the threshold size as needed
                cv2.drawContours(stretched_rgb, [contour], -1, (0, 255, 0), 2)  # Draw in green
                        # Calculate the moments
                M = cv2.moments(contour)
                if M["m00"] != 0:  # Prevent division by zero
                    # Calculate centroid
                    cx = int(M["m10"] / M["m00"])  # x-coordinate of center
                    cy = int(M["m01"] / M["m00"])  # y-coordinate of center

                    # Draw the centroid on the image
                    cv2.circle(stretched_rgb, (cx, cy), 3, (0, 255, 255), -1)  # Draw in red

        image1_rgb = cv2.cvtColor(stretched_rgb, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        image3_rgb = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
        image4_rgb = cv2.cvtColor(close, cv2.COLOR_BGR2RGB)
        image5_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        plt.subplot(1, 3, 1)
        plt.imshow(image1_rgb)
        plt.axis("off")  
        plt.subplot(1, 3, 2)
        plt.imshow(image4_rgb)
        plt.axis("off")  
        plt.subplot(1, 3, 3)
        plt.imshow(image5_rgb)
        plt.axis("off")  
        # plt.subplot(2, 2, 4)
        # plt.imshow(image3_rgb)
        # plt.axis("off")  
        plt.show()

        # color = ('b','g','r')
        # for i,col in enumerate(color):
        #     histr = cv2.calcHist([result2],[i],None,[256],[0,256])
        #     plt.plot(histr,color = col)
        #     plt.xlim([0,256])
        # plt.show()

        # base_filename = os.path.splitext(filename)[0]  
        # output_path = os.path.join(output_directory, f"{base_filename}_White_Pieces.jpg")
        # cv2.imwrite(output_path, final_result)


import cv2
import numpy as np
import os

# Directory containing the images
image_directory = "./Dataset/test"  # Replace with the path to your directory
output_directory = "./Output"  # Replace with your output directory

# Step 1: Process all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp")):  # Supported image formats
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not open or find the image: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grey", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows

        blurred = cv2.GaussianBlur(gray, (13, 13), 0  )

        # Step 4: Subtract the blurred image from the original
        # Resize original image to match the size of the downsampled image
        subtracted = cv2.absdiff(gray, blurred)
        cv2.imshow("Subtracted", subtracted)
        cv2.waitKey(0)
        cv2.destroyAllWindows

        # Step 5: Apply morphological closing operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Define the kernel size
        closed = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Morphological Closing", closed)
        cv2.waitKey(0)
        cv2.destroyAllWindows

        ret, thresh1 = cv2.threshold(closed,30,255,cv2.THRESH_BINARY)
        # blurred_binary = cv2.GaussianBlur(thresh1, (5, 5), 0  )
        cv2.imshow("Binarized", thresh1)
        cv2.waitKey(0)
        cv2.destroyAllWindows    

        # Step 8: Use Hough Line Transform to find lines
        lines = cv2.HoughLinesP(thresh1, 1, np.pi / 180, 100, minLineLength=75, maxLineGap=10)

        # Step 9: Draw the lines on the image
        result = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows   


    

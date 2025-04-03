import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

# Function to compute intersection points between lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:  # Parallel lines
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return int(px), int(py)

def fit_line(points):
    """Fits a line using least squares regression and returns slope & intercept (y = mx + b)."""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T  # Create design matrix
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve for m (slope) and b (intercept)
    return m, b

def line_intersection2(m1, b1, m2, b2):
    """Finds intersection of two lines given by y = m1*x + b1 and y = m2*x + b2."""
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return np.array([x, y])

def find_outer_perimeter(image, points):
    points = np.array(points)

    # Find extreme points along each boundary
    left_points = points[np.argsort(points[:, 0])][1:8]   
    right_points = points[np.argsort(points[:, 0])][-5:]
    top_points = points[np.argsort(points[:, 1])][:6] 
    if(points[np.argsort(points[:, 1])][-1][1] - points[np.argsort(points[:, 1])][-12][1] < 50):
        bottom_points = points[np.argsort(points[:, 1])][-16:-11]
    else:
        bottom_points = points[np.argsort(points[:, 1])][-7:]

    # Fit lines to each boundary
    m_left, b_left = fit_line(left_points)
    m_right, b_right = fit_line(right_points)
    m_top, b_top = fit_line(top_points)
    m_bottom, b_bottom = fit_line(bottom_points)

    # Compute intersections to get four corners
    top_left = line_intersection2(m_left, b_left, m_top, b_top)
    top_right = line_intersection2(m_right, b_right, m_top, b_top)
    bottom_right = line_intersection2(m_right, b_right, m_bottom, b_bottom)
    bottom_left = line_intersection2(m_left, b_left, m_bottom, b_bottom)

    for corner in left_points:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    for corner in right_points:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 255), -1)

    for corner in top_points:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 255), -1)

    for corner in bottom_points:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

    cv2.imshow("Sides", image)
    cv2.waitKey(0)

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners, [[m_left, b_left], [m_right, b_right], [m_top, b_top], [m_bottom, b_bottom]]

def warp_perspective(image, corners, output_size=(416, 416)):
    # Define the destination points (a perfect square)
    width, height = output_size
    offset = 100
    dst_points = np.array([
        [0+offset, 0+offset],           # Top-left corner
        [width-offset, 0+offset],     # Top-right corner
        [width-offset, height-offset], # Bottom-right corner
        [0+offset, height-offset],     # Bottom-left corner
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Apply the transformation
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    return warped_image

def get_base_board():
    image_path = "./Dataset/test/e0d38d159ad3a801d0304d7e275812cc_jpg.rf.0cd06a940ccc9894109d83792535e3eb.jpg"
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    subtracted = cv2.absdiff(gray, blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, thresh1 = cv2.threshold(closed, 20, 255, cv2.THRESH_BINARY)

    # Hough Line Transform to find lines
    lines = cv2.HoughLinesP(thresh1, 1, np.pi / 180, 150, minLineLength=225, maxLineGap=100)

    # List to store intersection points
    intersection_points = []

    # Find intersections between all detected lines
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = line_intersection(lines[i][0], lines[j][0])
                if intersection:
                    intersection_points.append(intersection)

    # Cluster intersection points using DBSCAN
    if intersection_points:
        points = np.array(intersection_points)
        clustering = DBSCAN(eps=10, min_samples=2).fit(points)
        cluster_centers = []
        for label in set(clustering.labels_):
            if label != -1:  # Ignore noise points
                cluster = points[clustering.labels_ == label]
                cluster_centers.append(np.mean(cluster, axis=0))

        # Sort the corners in a structured order (row by row)
        sorted_corners = sorted(cluster_centers, key=lambda x: (x[1], x[0]))  # Sort by y, then x

        # Validate corners to ensure they are within the image bounds
        height, width = gray.shape
        valid_corners = [point for point in sorted_corners if 0 <= point[0] < width and 0 <= point[1] < 375]
        valid_corners = np.array(valid_corners, dtype=np.float32)

        if len(valid_corners) > 0:
            # Refine corners to subpixel accuracy
            refined_corners = cv2.cornerSubPix(
                gray,
                valid_corners,
                (5, 5),
                (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )

            corners, lines = find_outer_perimeter(image, refined_corners)

            corners_int = np.array(corners).astype(int)

            # Draw the box by connecting the corners
            for i in range(len(corners_int)):
                # Get the current corner and the next corner
                start_point = tuple(corners_int[i])
                end_point = tuple(corners_int[(i + 1) % len(corners_int)])  # Wrap around to the first corner

                # Draw the line between the two corners
                cv2.line(image, start_point, end_point, color=(255, 0, 255), thickness=2)  # Red line

            for corner in refined_corners:
                x, y = corner.ravel()
                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

            cv2.imshow("Empty Board", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows

            return image, corners

# Directory containing the images
image_directory = "./Dataset/test"  # Replace with the path to your directory
output_directory = "./Output"  # Replace with your output directory

base_image, base_corners = get_base_board()
base_gray_image = gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

# Process all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp")):  # Supported image formats
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        image_copy = image.copy()
        
        if image is None:
            print(f"Could not open or find the image: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        subtracted = cv2.absdiff(gray, blurred)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        closed = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=1)

        cv2.imshow("Edges", closed)
        cv2.waitKey(0)

        _, thresh1 = cv2.threshold(closed, 20, 255, cv2.THRESH_BINARY)

        # Hough Line Transform to find lines
        lines = cv2.HoughLinesP(thresh1, 1, np.pi / 180, 150, minLineLength=225, maxLineGap=100)

        # List to store intersection points
        intersection_points = []

        # Find intersections between all detected lines
        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    intersection = line_intersection(lines[i][0], lines[j][0])
                    if intersection:
                        intersection_points.append(intersection)

        # Cluster intersection points using DBSCAN
        if intersection_points:
            points = np.array(intersection_points)
            clustering = DBSCAN(eps=10, min_samples=2).fit(points)
            cluster_centers = []
            for label in set(clustering.labels_):
                if label != -1:  # Ignore noise points
                    cluster = points[clustering.labels_ == label]
                    cluster_centers.append(np.mean(cluster, axis=0))

            # Sort the corners in a structured order (row by row)
            sorted_corners = sorted(cluster_centers, key=lambda x: (x[1], x[0]))  # Sort by y, then x

            # Validate corners to ensure they are within the image bounds
            height, width = gray.shape
            valid_corners = [point for point in sorted_corners if 0 <= point[0] < width and 0 <= point[1] < 375]
            valid_corners = np.array(valid_corners, dtype=np.float32)

            if len(valid_corners) > 0:
                # Refine corners to subpixel accuracy
                refined_corners = cv2.cornerSubPix(
                    gray,
                    valid_corners,
                    (5, 5),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )

                # Draw the detected and refined corners
                # for corner in refined_corners:
                #     x, y = corner.ravel()
                #     cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

                corners, lines = find_outer_perimeter(image, refined_corners)

                '''DRAWING OUTER PERIMETER'''
                
                # corners_int = np.array(corners).astype(int)
                # # Draw the box by connecting the corners
                # for i in range(len(corners_int)):
                #     # Get the current corner and the next corner
                #     start_point = tuple(corners_int[i])
                #     end_point = tuple(corners_int[(i + 1) % len(corners_int)])  # Wrap around to the first corner

                #     # Draw the line between the two corners
                #     cv2.line(image, start_point, end_point, color=(255, 0, 255), thickness=2)  # Red line

                '''FINDING BLACK PIECES'''

                # Threshold the image (binary inverse to detect dark regions)
                _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)        

                # Find contours of the blobs
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                '''DRAWING GRID POINTS'''

                # # Draw contours on the original image
                # for contour in contours:
                #     # Optionally filter by blob size
                #     area = cv2.contourArea(contour)
                #     if area > 100:  # Adjust the threshold size as needed
                #         cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw in green
                #                 # Calculate the moments
                #         M = cv2.moments(contour)
                #         if M["m00"] != 0:  # Prevent division by zero
                #             # Calculate centroid
                #             cx = int(M["m10"] / M["m00"])  # x-coordinate of center
                #             cy = int(M["m01"] / M["m00"])  # y-coordinate of center

                #             # Draw the centroid on the image
                #             cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)  # Draw in red

                '''FINDING WHITE PEICES'''

                # Compute the perspective transform matrix
                transform_matrix = cv2.getPerspectiveTransform(base_corners, corners)

                # Apply the transformation
                warped_base_image = cv2.warpPerspective(base_image, transform_matrix, image.shape[:2])
                foreground = cv2.absdiff(gray, base_gray_image)
                cv2.imshow("Background Subtraction", foreground)     
                
                grey_boosted = cv2.addWeighted(gray, 1.5, np.zeros(gray.shape, gray.dtype), 0, 0) 
                cv2.imshow("Grey Boosted", grey_boosted)  
                _, binary_mask = cv2.threshold(grey_boosted,150,255,cv2.THRESH_BINARY) 
                masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
                cv2.imshow("Binary Mask", masked_image)  
                color = ('b','g','r')
                for i,col in enumerate(color):
                    histr = cv2.calcHist([masked_image],[i],None,[256],[0,256])
                    plt.plot(histr,color = col)
                    plt.xlim([0,256])
                plt.show()

                # Create a binary mask where green values are above 214
                _, green_mask = cv2.threshold(masked_image[:,:,1], 214, 255, cv2.THRESH_BINARY)
                cv2.imshow("Green Mask", green_mask)  
                # Apply the mask to the original image
                blue_masked_image = cv2.bitwise_and(masked_image, masked_image, mask=green_mask)

                adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,2)
                cv2.imshow("Adaptive Threshold", adaptive_thresh)                

                warp = warp_perspective(image, corners)
                cv2.imshow("Warped", warp)
            else:
                print("No valid corners to refine!")
        else:
            print("No intersections found!")

        # Display the result
        cv2.imshow("Detected Corners", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Detected Corners")
        cv2.destroyWindow("Warped")
        cv2.destroyWindow("Background Subtraction")
        cv2.destroyWindow("Sides")
        cv2.destroyWindow("Grey Boosted")
        cv2.destroyWindow("Binary Mask")
        cv2.destroyWindow("Adaptive Threshold")
        cv2.destroyWindow("Edges")
        cv2.destroyWindow("Green Mask")

#unsharpening
#contrast stretched
#adaptive thresholding
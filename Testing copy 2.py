import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

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

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed at p2 by lines p1-p2 and p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_slope(p1, p2):
    """Calculate the slope of the line formed by two points"""
    if p2[0] - p1[0] == 0:
        return float('inf')  # Avoid division by zero
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def is_trapezoid(quad):
    """Check if the given 4 points form a trapezoid with specified angle conditions"""
    points = np.array(quad, dtype=np.float32)
    points = sorted(points, key=lambda p: p[1])  # Sort by y-coordinate (ascending)
    
    top_points = sorted(points[:2], key=lambda p: p[0])  # Sort top two by x-coordinate
    bottom_points = sorted(points[2:], key=lambda p: p[0])  # Sort bottom two by x-coordinate
    
    top_left, top_right = top_points
    bottom_left, bottom_right = bottom_points
    
    # Compute angles at each corner
    angle_tl = calculate_angle(bottom_left, top_left, top_right)
    angle_tr = calculate_angle(top_left, top_right, bottom_right)
    angle_bl = calculate_angle(top_left, bottom_left, bottom_right)
    angle_br = calculate_angle(bottom_left, bottom_right, top_right)

    # Compute slopes for top and bottom edges
    top_slope = calculate_slope(top_left, top_right)
    bottom_slope = calculate_slope(bottom_left, bottom_right)
    left_slope = calculate_slope(top_left, bottom_left)
    right_slope = calculate_slope(top_right, bottom_right)

    # Define conditions

    top_condition = True
    bottom_condition = True
    ratio_condition = True
    horizontal_slope_condition = True
    vertical_slope_condition = True
    trap_condition = True

    top_condition = 87 <= angle_tl <= 103 and 87 <= angle_tr <= 103
    bottom_condition = 77 <= angle_bl <= 93 and 77 <= angle_br <= 93
    # ratio_condition = (angle_tl + angle_tr) / (angle_bl + angle_br) > 1
    horizontal_slope_condition = abs(top_slope - bottom_slope) < 0.1 and abs(top_slope) < 0.1 and abs(bottom_slope) < 0.1
    # vertical_slope_condition = left_slope - right_slope < 0 
    trap_condition = (angle_bl < angle_tl and angle_bl < angle_tr) and (angle_br < angle_tl and angle_br < angle_tr)
    
    overall_condition = top_condition and bottom_condition and horizontal_slope_condition and vertical_slope_condition and trap_condition and ratio_condition

    corner_info = None
    slope_info = None

    if (overall_condition):
        # Visualize the points
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
        labels = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
        angles = [angle_tl, angle_tr, angle_bl, angle_br]

        corner_info = zip([top_left, top_right, bottom_left, bottom_right], colors, labels, angles)
        slope_info = [top_slope, bottom_slope, left_slope, right_slope]
    
    return overall_condition, corner_info, slope_info

def find_largest_quadrilateral(points):
    # Ensure points are in the correct format
    points = np.array(points, dtype=np.float32)

    # Calculate the convex hull of the points
    hull = cv2.convexHull(points)

    # Otherwise, iterate to find the largest valid 4-sided polygon
    largest_quadrilateral = None
    max_area = 0

    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            for k in range(j + 1, len(hull)):
                for l in range(k + 1, len(hull)):
                    # Form a quadrilateral with 4 points
                    quad = np.array([hull[i][0], hull[j][0], hull[k][0], hull[l][0]], dtype=np.float32)

                    # Calculate the area
                    area = cv2.contourArea(quad)

                    # Check if it's a valid quadrilateral with angles near 90°
                    if area > max_area:
                        is_trap, corner_info, slope_info = is_trapezoid(quad)
                        if is_trap:
                            max_area = area
                            largest_quadrilateral = quad

    return largest_quadrilateral, corner_info, slope_info

def warp_perspective(image, corners, output_size=(500, 500)):
    # Define the destination points (a perfect square)
    width, height = output_size
    dst_points = np.array([
        [0, 0],           # Top-left corner
        [width-1, 0],     # Top-right corner
        [width-1, height-1], # Bottom-right corner
        [0, height-1]     # Bottom-left corner
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Apply the transformation
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    return warped_image

# Directory containing the images
image_directory = "./Dataset/test"  # Replace with the path to your directory
output_directory = "./Output"  # Replace with your output directory

# Process all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp")):  # Supported image formats
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not open or find the image: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        subtracted = cv2.absdiff(gray, blurred)

        cv2.imshow("Subtracted", subtracted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        closed = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=1)

        cv2.imshow("Closed", closed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        _, thresh1 = cv2.threshold(closed, 20, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Binary", thresh1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                for corner in refined_corners:
                    x, y = corner.ravel()
                    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

                # largest_quad, corner_info, slope_info = find_largest_quadrilateral(refined_corners)

                # if largest_quad is not None and corner_info is not None and slope_info is not None:
                #     # Draw the quadrilateral on the image
                #     largest_quad = largest_quad.astype(int)
                #     cv2.polylines(image, [largest_quad], isClosed=True, color=(0, 255, 0), thickness=2)

                #     for (point, color, label, angle) in corner_info:
                #         x, y =  map(int, point)
                #         cv2.circle(image, (x, y), 5, color, -1)
                #         print(f"{label}: {point}, Angle: {angle:.2f} degrees, Color: {color}")

                #     print("Top slope: ", slope_info[0])
                #     print("Bottom slope: ", slope_info[1])
                #     print("Left slope: ", slope_info[2])
                #     print("Right slope: ", slope_info[3])
                else:
                    print("No valid quadrilateral found!")
            else:
                print("No valid corners to refine!")
        else:
            print("No intersections found!")

        # Display the result
        cv2.imshow("Detected Corners", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

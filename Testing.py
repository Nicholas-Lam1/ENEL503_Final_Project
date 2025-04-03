import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

# Function to approximate a 10x10 grid
def approximate_chessboard(found_corners, board_size=(10, 10)):
    rows, cols = board_size
    found_corners = np.array(found_corners, dtype=np.float32)
    
    # Fit a bounding quadrilateral to the found corners using convex hull
    rect = cv2.minAreaRect(found_corners)
    box = cv2.boxPoints(rect)  # Get the box corners
    box = np.array(sorted(box, key=lambda x: (x[1], x[0])), dtype=np.float32)

    # Define the target grid points (perfect 10x10 grid)
    target_grid = np.array([[x, y] for y in range(rows) for x in range(cols)], dtype=np.float32)
    
    # Perspective transform to map the approximate quadrilateral to a grid
    matrix = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]]))
    transformed_corners = cv2.perspectiveTransform(found_corners[None, :, :], matrix)[0]

    # Find the nearest neighbors to the grid
    nearest_grid_points = []
    for corner in transformed_corners:
        grid_x, grid_y = np.round(corner).astype(int)
        if 0 <= grid_x < cols and 0 <= grid_y < rows:
            nearest_grid_points.append((grid_x, grid_y))

    # Transform grid points back to image space
    inverse_matrix = np.linalg.inv(matrix)
    grid_points = cv2.perspectiveTransform(np.array([target_grid], dtype=np.float32), inverse_matrix)[0]

    return grid_points

# Main function to process a single image
def process_chessboard_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours to detect corners
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_corners = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for point in approx:
            found_corners.append(point[0])

    # Cluster corners to reduce noise
    clustering = DBSCAN(eps=15, min_samples=2).fit(found_corners)
    clustered_corners = []
    for label in set(clustering.labels_):
        if label != -1:  # Ignore noise points
            cluster = np.array(found_corners)[clustering.labels_ == label]
            clustered_corners.append(np.mean(cluster, axis=0))

    # Approximate the 10x10 grid
    chessboard_corners = approximate_chessboard(clustered_corners)

    # Draw the grid on the image
    result = image.copy()
    for corner in chessboard_corners:
        x, y = np.round(corner).astype(int)
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow("Chessboard Corners", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Directory containing the images
image_directory = "./Dataset/test"  # Replace with the path to your directory
output_directory = "./Output"  # Replace with your output directory

# Step 1: Process all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp")):  # Supported image formats
        image_path = os.path.join(image_directory, filename)
        process_chessboard_image(image_path)

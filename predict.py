import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:  
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return int(px), int(py)

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T  
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  
    return m, b

def line_intersection2(m1, b1, m2, b2):
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return np.array([x, y])

def find_outer_perimeter(image, points):
    points = np.array(points)

    left_points = points[np.argsort(points[:, 0])][1:8]   
    right_points = points[np.argsort(points[:, 0])][-5:]
    top_points = points[np.argsort(points[:, 1])][:6] 
    if(points[np.argsort(points[:, 1])][-1][1] - points[np.argsort(points[:, 1])][-12][1] < 50):
        bottom_points = points[np.argsort(points[:, 1])][-16:-11]
    else:
        bottom_points = points[np.argsort(points[:, 1])][-7:]

    m_left, b_left = fit_line(left_points)
    m_right, b_right = fit_line(right_points)
    m_top, b_top = fit_line(top_points)
    m_bottom, b_bottom = fit_line(bottom_points)

    top_left = line_intersection2(m_left, b_left, m_top, b_top)
    top_right = line_intersection2(m_right, b_right, m_top, b_top)
    bottom_right = line_intersection2(m_right, b_right, m_bottom, b_bottom)
    bottom_left = line_intersection2(m_left, b_left, m_bottom, b_bottom)

    # for corner in left_points:
    #     x, y = corner.ravel()
    #     cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # for corner in right_points:
    #     x, y = corner.ravel()
    #     cv2.circle(image, (int(x), int(y)), 2, (0, 255, 255), -1)

    # for corner in top_points:
    #     x, y = corner.ravel()
    #     cv2.circle(image, (int(x), int(y)), 2, (255, 0, 255), -1)

    # for corner in bottom_points:
    #     x, y = corner.ravel()
    #     cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

    # cv2.imshow("Sides", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners, [[m_left, b_left], [m_right, b_right], [m_top, b_top], [m_bottom, b_bottom]]

def warp_perspective(image, corners, piece_loc, output_size=(416, 416)):
    width, height = output_size
    offset = 100
    dst_points = np.array([
        [0+offset, 0+offset],         
        [width-offset, 0+offset],     
        [width-offset, height-offset],
        [0+offset, height-offset],     
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    piece_loc = np.array(piece_loc, dtype='float32').reshape(-1, 1, 2)

    transformed_loc = cv2.perspectiveTransform(piece_loc, transform_matrix)
    transformed_loc = [tuple(pt[0]) for pt in transformed_loc]

    return warped_image, transformed_loc


image_directory = "./datasets/chess/test/images/"  

model = YOLO("runs/detect/yolov8_chess/weights/best.pt")

for file_name in os.listdir(image_directory):
    if file_name.endswith((".jpg", ".png", ".jpeg", ".bmp")):  
        # Get image
        image_path = image_directory + file_name
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            exit()

        # Find chess pieces and co-ordinates
        results = model(image_path)
        piece_loc = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = y2 - 10

                piece_loc.append([center_x, center_y])

        if len(piece_loc) == 0:
            print(f"Could not find any pieces: {image_path}")
            continue

        # Find lines of chessboard, calculate intesections, find corners, and warp board
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        subtracted = cv2.absdiff(gray, blurred)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, thresh1 = cv2.threshold(closed, 20, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(thresh1, 1, np.pi / 180, 150, minLineLength=225, maxLineGap=100)

        intersection_points = []

        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    intersection = line_intersection(lines[i][0], lines[j][0])
                    if intersection:
                        intersection_points.append(intersection)

        if intersection_points:
            points = np.array(intersection_points)
            clustering = DBSCAN(eps=10, min_samples=2).fit(points)
            cluster_centers = []
            for label in set(clustering.labels_):
                if label != -1: 
                    cluster = points[clustering.labels_ == label]
                    cluster_centers.append(np.mean(cluster, axis=0))

            sorted_corners = sorted(cluster_centers, key=lambda x: (x[1], x[0]))  

            height, width = gray.shape
            valid_corners = [point for point in sorted_corners if 0 <= point[0] < width and 0 <= point[1] < 375]
            valid_corners = np.array(valid_corners, dtype=np.float32)

            if len(valid_corners) > 0:
                refined_corners = cv2.cornerSubPix(
                    gray,
                    valid_corners,
                    (5, 5),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )

                corners, lines = find_outer_perimeter(image, refined_corners)

                warp, warp_piece_loc = warp_perspective(image, corners, piece_loc)

                for piece in warp_piece_loc:
                    piece_int = tuple(map(int, piece)) 
                    cv2.circle(warp, piece_int, radius=2, color=(0, 0, 255), thickness=-1)
            else:
                print("No valid corners found")
        else:
            print("No valid intersections found")

        # cv2.imshow("Output", warp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            
        output_path = "./output/result_" + file_name
        cv2.imwrite(output_path, warp)

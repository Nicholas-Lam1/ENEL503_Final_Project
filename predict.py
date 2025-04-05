import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

# Mapping from YOLO class ID (from data.yaml) to FEN notation
piece_mapping = {
    0: 'b',  # bishop (assumed black)
    1: 'b',  # black-bishop
    2: 'k',  # black-king
    3: 'n',  # black-knight
    4: 'p',  # black-pawn
    5: 'q',  # black-queen
    6: 'r',  # black-rook
    7: 'B',  # white-bishop
    8: 'K',  # white-king
    9: 'N',  # white-knight
    10: 'P', # white-pawn
    11: 'Q', # white-queen
    12: 'R'  # white-rook
}

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
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners

def warp_perspective(image, corners, piece_loc, output_size=(416, 416)):
    width, height = output_size
    offset = 0 # adjust margin as needed
    dst_points = np.array([
        [offset, offset],
        [width-offset, offset],
        [width-offset, height-offset],
        [offset, height-offset],
    ], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))
    piece_loc = np.array(piece_loc, dtype='float32').reshape(-1, 1, 2)
    transformed_loc = cv2.perspectiveTransform(piece_loc, transform_matrix)
    transformed_loc = [tuple(pt[0]) for pt in transformed_loc]
    return warped_image, transformed_loc

def segment_board(warped_image):
    # Divide the warped board into an 8x8 grid
    height, width, _ = warped_image.shape
    cell_w = width // 8
    cell_h = height // 8
    grid = []
    for row in range(8):
        row_cells = []
        for col in range(8):
            x_start = col * cell_w
            y_start = row * cell_h
            row_cells.append((x_start, y_start, cell_w, cell_h))
        grid.append(row_cells)
    return grid

def map_pieces_to_grid(piece_detections, grid):
    # Create an empty 8x8 board (rows from 0 to 7 - top row corresponds to rank 8)
    board = [['' for _ in range(8)] for _ in range(8)]
    # Use grid cell size from the first cell 
    cell_w, cell_h = grid[0][0][2], grid[0][0][3]
    for det in piece_detections:
        x, y = det['warp_loc']
        col = int(x // cell_w)
        row = int(y // cell_h)
        if 0 <= row < 8 and 0 <= col < 8:
            board[row][col] = piece_mapping.get(det['class'], '?')
    return board

def generate_fen(board):
    # Convert the 8x8 board array into a FEN string
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    # FEN requires rows from rank 8 (top) to rank 1 (bottom)
    return "/".join(fen_rows[::-1])

image_directory = "./datasets/chess/test/images/"
output_directory = "./output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


model = YOLO("runs/detect/yolov8_chess/weights/best.pt")

for file_name in os.listdir(image_directory):
    if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        image_path = os.path.join(image_directory, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not open image: {image_path}")
            continue

        # Piece Detection and Classification
        results = model(image_path)
        piece_detections = []
        # For each detection result extract box coordinates and predicted class
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Extract predicted class
                class_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                center_x = (x1 + x2) // 2
                center_y = y2 - 10 
                piece_detections.append({'loc': [center_x, center_y], 'class': class_id})

        if not piece_detections:
            print(f"No pieces detected in {image_path}")
            continue

        # Prepare list of piece locations for perspective transformation
        original_piece_locs = [det['loc'] for det in piece_detections]

        # Chessboard Detection and Correction
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
                corners = find_outer_perimeter(image, refined_corners)
                # Warp board and transform piece coordinates
                warp, warp_piece_loc = warp_perspective(image, corners, original_piece_locs)
                # Update each detection with its warped coordinate
                for idx, det in enumerate(piece_detections):
                    det['warp_loc'] = warp_piece_loc[idx]

                # Square Segmentation
                grid = segment_board(warp)

                # FEN Mapping
                board = map_pieces_to_grid(piece_detections, grid)
                fen = generate_fen(board)
                print(f"FEN for {file_name}: {fen}")

                # Display grid and detections on warped image
                for row in grid:
                    for (x, y, w_cell, h_cell) in row:
                        cv2.rectangle(warp, (x, y), (x + w_cell, y + h_cell), (255, 0, 0), 1)
                for det in piece_detections:
                    pt = tuple(map(int, det['warp_loc']))
                    cv2.circle(warp, pt, 3, (0, 0, 255), -1)

                img_output_path = os.path.join(output_directory, "result_" + file_name)
                cv2.imwrite(img_output_path, warp)

                fen_output_path = os.path.join(output_directory, "result_" + os.path.splitext(file_name)[0] + ".txt")

                with open(fen_output_path, 'w') as fen_file:
                    fen_file.write(fen)
            else:
                print("No valid corners found for chessboard detection.")
        else:
            print("No intersections found for chessboard detection.")

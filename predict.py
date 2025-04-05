from ultralytics import YOLO
import cv2
import sys

model = YOLO("runs/detect/yolov8_chess/weights/best.pt")

file_name = 'b526b661a33ff481231d1342aff2a266_jpg.rf.287d21a885ec3abeb6da818a6a9cd05b.jpg'
image_path = "./datasets/chess/test/images/" + file_name
image = cv2.imread(image_path)

results = model(image_path)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = y2 - 10

        cv2.circle(image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

output_path = "./output/result_" + file_name
cv2.imwrite(output_path, image)

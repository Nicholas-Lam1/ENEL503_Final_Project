from ultralytics import YOLO
import sys

model = YOLO(
    "runs/detect/yolov8_chess/weights/best.pt")

file_name = 'b526b661a33ff481231d1342aff2a266_jpg.rf.287d21a885ec3abeb6da818a6a9cd05b.jpg'
results = model(
    "datasets/chess/test/images/" + file_name,
    save=True,
    exist_ok=True
)

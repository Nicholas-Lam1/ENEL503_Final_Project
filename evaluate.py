from ultralytics import YOLO

# best checkpoint
model = YOLO(
    "runs/detect/yolov8_chess/weights/best.pt")
metrics = model.val()  # runs validation on the validation set

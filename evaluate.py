from ultralytics import YOLO

# best checkpoint
model = YOLO(
    "runs/detect/yolov8_chess/weights/best.pt")
metrics = model.val(exist_ok=True)  # runs validation on the validation set

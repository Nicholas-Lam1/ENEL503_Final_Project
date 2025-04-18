import os
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.to('cuda')

    # Making an absolute path since relative path wasn't working for some reason
    absolute_path = os.path.join(os.getcwd(), 'datasets/chess/data.yaml')

    model.train(
        data=absolute_path,
        epochs=25,  
        imgsz=640,
        batch=16,
        name="yolov8_chess",
        exist_ok=True
    )

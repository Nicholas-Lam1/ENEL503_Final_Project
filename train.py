from ultralytics import YOLO
import os

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # Making an absolute path since relative path wasn't working for some reason
    absolute_path = os.path.join(os.getcwd(), 'datasets/chess/data.yaml')

    model.train(
        # Dunno why but relative path doesn't work here
        data=absolute_path,
        epochs=4,  # Should probably increase
        imgsz=640,
        batch=16,
        name="yolov8_chess",
        exist_ok=True
    )

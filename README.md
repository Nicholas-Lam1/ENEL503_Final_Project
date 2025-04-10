# PieceWise - Chessboard Image to FEN Converter

---

## Instructions to Run

***Note:***
Unfortunately, YOLO seems to utilize absolute paths when training a model. As such, simply running the code will not work. A model must be trained first to generate the model and local files associated with it. This may be a time consuming task and may not be viable during testing.

### **Commands to Run:**

If you would still like to run the program, please enter these commands in the console, in order:

`python train.py`

`python predict.py`

`python view.py`

### Alternative Testing:

If testing the program must be done, but in a timely manner, please make the following change in train.py:

```python
model.train(
    data=absolute_path,
    epochs=25,  #Please lower this value to 5
    imgsz=640,
    batch=16,
    name="yolov8_chess_detect",
    exist_ok=True
)
```

Additionally, please delete the `runs` folder. It contains the 25 epoch - standard weight model we trained and will not be replaced by new training, as the best model is kept and utilized.

Once these changes have been made, please follow the instructions explained in `Commands to Run`. Training should be quicker, however, the results will not be reflective of the performance we were able to achieve.


Thank you so much for the semester, have a great summer!

from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict("input_videos/08fd33_4.mp4",save = True)
print(results[0])
print("="*30)
for box in  results[0].boxes:       # if we take a look at the output video we can see that
    print(box)                      # the model is having some hard detecting the ball.
                                    # therefore, we have to fine-tune it in this case we use the data from roboflow.















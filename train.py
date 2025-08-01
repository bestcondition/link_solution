from ultralytics import YOLO

model = YOLO('yolo11l.pt')
model.train(
    data='data.yaml',
    epochs=200,
    batch=1,
    imgsz=1024,
)

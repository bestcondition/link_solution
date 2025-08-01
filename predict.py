from ultralytics import YOLO

model = YOLO('best.pt')
model('test.jpg', save=True)

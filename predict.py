from ultralytics import YOLO

model = YOLO('best.pt')
model('images', save=True)

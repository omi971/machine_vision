import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
resuts = model('image/1.png', show=True)

cv2.waitKey(0)

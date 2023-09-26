import cv2
from ultralytics import YOLO

# img = cv2.imread('image/2.png', 1)

# cv2.imshow("Sample", img)
# cv2.waitKey(5000)

model = YOLO('../yolo_weights/yolov8n.pt')
results = model('../image/3.png', show=True)

cv2.waitKey(0)

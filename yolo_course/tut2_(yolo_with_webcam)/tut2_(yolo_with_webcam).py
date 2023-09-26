from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../yolo_weights/yolov8n.pt')

prev_frame_time = 0
new_frame_time = 0

while True:

    # For FPS Counting
    new_frame_time = time.time()

    ret, img = cap.read()
    results = model(img, stream=True)
    

    # FPS Measurements
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("Detection FPS: ", fps)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Webcam Stream', img)

    # Quit window function
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

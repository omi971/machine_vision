from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../yolo_weights/yolov8n.pt')

while True:
    ret, frame = cap.read()

    results = model(frame, stream=True, show=True)
    cv2.imshow('Webcam Stream', frame)
    cv2.waitKey(1)
# cv2.imshow('Image', frame)
# cv2.waitKey(0)


# cap.release()
# cv2.destroyAllWindows()

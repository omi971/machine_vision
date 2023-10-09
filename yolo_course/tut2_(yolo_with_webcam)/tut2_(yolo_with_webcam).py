# yolo running in cpu

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("..\\videos\\cars.mp4")
cap.set(3, 1280)  # Frame Width
cap.set(4, 720)  # Frame Height

model = YOLO('../yolo_weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

while True:

    # For FPS Counting
    new_frame_time = time.time()

    ret, img = cap.read()
    results = model(img, stream=True)
    print('Results: ', results)

    for r in results:
        print('r:', r)
        boxes = r.boxes  # ??
        print('boxes', boxes)
        for box in boxes:
            # for normal rectangle box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # for cvzone lib
            # x, y, w, h = box.xywh[0]
            # bbox = int(x), int(y), int(w), int(h)

            # coordinates of bounding box
            # print(f'x1: {x1}, y2: {y2}, x2: {x2}, y2: {y2}')
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(255, 255, 255))

            # Detection Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence", conf)
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))  # review again ** max func

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

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

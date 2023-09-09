from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../yolo_weights/yolov8n.pt')

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

while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)
    print(results)

    # Drawing Bounding boxes for the detected object
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, w, h = box.xywh[0]

            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            print(f'Coordinates --> x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')

            # For cvzone library we need width and height
            cvzone.cornerRect(frame, bbox)

            conf = math.ceil(box.conf[0] * 100) / 100
            print('Confidence', conf)

            # Giving Error for the library-problem
            cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(0, y1)))
    cv2.imshow('Webcam Stream', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

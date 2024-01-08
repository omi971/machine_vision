# yolo running in cpu

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("..\\videos\\cars.mp4")
cap.set(3, 1280)  # Frame Width
cap.set(4, 720)  # Frame Height

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f'Frame Width: {frame_width}')

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Frame Height: {frame_height}')

# Calculate center coordinates
fcenter_x = frame_width // 2
fcenter_y = frame_height // 2

# model = YOLO('../yolo_weights/yolov8n.pt')  # This is light model
model = YOLO('../yolo_weights/yolov8l.pt')  # This is large model

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
    # print('Results: ', results)

    # Draw a vertical line (green)
    cv2.line(img, (fcenter_x, 0), (fcenter_x, frame_height), (0, 255, 0), 2)

    # Draw a horizontal line (blue)
    cv2.line(img, (0, fcenter_y), (frame_width, fcenter_y), (0, 255, 0), 2)

    # Frame Center threshold box
    rect_width = 250
    rect_height = 250

    # Calculate top-left and bottom-right coordinates of the rectangle
    top_left_x = fcenter_x - rect_width // 2
    top_left_y = fcenter_y - rect_height // 2
    bottom_right_x = fcenter_x + rect_width // 2
    bottom_right_y = fcenter_y + rect_height // 2

    # Calculate top-right and bottom-left coordinates
    top_right_x = bottom_right_x
    top_right_y = top_left_y
    bottom_left_x = top_left_x
    bottom_left_y = bottom_right_y

    threshold = 100
    # Draw the rectangle
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    for r in results:
        # print('r:', r)
        boxes = r.boxes  # ??
        # print('boxes', boxes)
        for box in boxes:
            # for normal rectangle box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # for cvzone lib
            # x, y, w, h = box.xywh[0]
            # bbox = int(x), int(y), int(w), int(h)

            # coordinates of bounding box
            # print(f'x1: {x1}, y2: {y2}, x2: {x2}, y2: {y2}')
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1

            # Detection Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print("Confidence", conf)
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))  # review again ** max func

            # Class Name
            cls = int(box.cls[0])  # index of the classname list
            currentClass = classNames[cls]  # class name

            # if currentClass == "motorbike": cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 255),
            # colorR=(255, 255, 255), l=15) cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35,
            # y1)), scale=1, thickness=1, offset=3) if detects car it will name box and show

            # if currentClass == "car" or currentClass == "motorbike" or currentClass == "truck" or currentClass == "bus" and conf > 0.3:
            #     cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(255, 255, 255), l=15)
            #     cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

            if currentClass == "fork" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(255, 255, 255), l=15)
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                   offset=3)

                # find the center of the bounding box
                bb_center_x = (x1 + x2) // 2
                bb_center_y = (y1 + y2) // 2

                # Draw horizontal line passing through the center
                cv2.line(img, (0, bb_center_y), (img.shape[1], bb_center_y), (255, 0, 0), 2)  # Green color

                # Draw vertical line passing through the center
                cv2.line(img, (bb_center_x, 0), (bb_center_x, img.shape[0]), (255, 0, 0), 2)  # Blue color

                # Draw a line from the bounding box center to the frame center
                cv2.line(img, (fcenter_x, fcenter_y), (bb_center_x, bb_center_y), (255, 0, 0), 2)  # Blue color

                # Check if the bounding box is inside the rectangle
                if (top_left_x < bb_center_x < bottom_right_x) and (top_left_y < bb_center_y < bottom_right_y):
                    temp = 'Bounding box is inside the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print('GO FORWARD')
                elif bb_center_x > bottom_right_x:
                    temp = 'Bounding box is to the right of the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print('GO LEFT')
                elif bb_center_x < top_left_x:
                    temp = 'Bounding box is to the left of the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print('GO RIGHT')

                # Don't need this (safety er jonno rekhe disi)
                else:
                    temp = 'Bounding box is neither inside nor to the sides of the rectangle'
                    print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # FPS Measurements
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print("Detection FPS: ", fps)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow('Webcam Stream', img)

    # Quit window function
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

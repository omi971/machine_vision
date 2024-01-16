# yolo running in cpu

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)

# Setting the frame resolution to 1280x720p
cap.set(3, 1280)  # Frame Width
cap.set(4, 720)  # Frame Height

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f'Frame Width: {frame_width}')

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Frame Height: {frame_height}')

# Calculate center coordinates of the frame
fcenter_x, fcenter_y = (frame_width // 2), (frame_height // 2)

model = YOLO('../yolo_weights/yolov8n.pt')  # This is nano model
# model = YOLO('../yolo_weights/yolov8s.pt')  # This is small modell
# model = YOLO('../yolo_weights/yolov8l.pt')  # This is large model

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

# For fps calculation value initialization
prev_frame_time = 0
new_frame_time = 0

while True:

    # For FPS Counting
    new_frame_time = time.time()

    ret, img = cap.read()
    if not ret:
        break
    results = model(img, stream=True)
    # print('Results: ', results)

    # Draw a vertical line (green)
    cv2.line(img, (fcenter_x, 0), (fcenter_x, frame_height), (0, 255, 0), 2)

    # Draw a horizontal line (green)
    cv2.line(img, (0, fcenter_y), (frame_width, fcenter_y), (0, 255, 0), 2)

    # Frame Center threshold box
    threshold = 250
    rect_width, rect_height = threshold, threshold

    # Calculate top-left and bottom-right coordinates of the rectangle
    top_left_x = fcenter_x - rect_width // 2
    top_left_y = fcenter_y - rect_height // 2
    bottom_right_x = fcenter_x + rect_width // 2
    bottom_right_y = fcenter_y + rect_height // 2

    # Draw the threshold rectangle
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    for r in results:
        # print(f"results: {results}")
        # print('r:', r)
        boxes = r.boxes  # ??
        # print('boxes', boxes)
        for box in boxes:
            # for normal (top-left, bottom-right) rectangle box coordinates
            # print(dir(box))
            print(box.xyxy[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # calculating bb width heights for easier calculation

            # Detection Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print("Confidence", conf)
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))  # review again ** max func

            # Class Name
            cls = int(box.cls[0])  # index of the class name list
            currentClass = classNames[cls]  # class name

            # if currentClass == "car" or currentClass == "motorbike" or currentClass == "truck" or currentClass ==
            # "bus" and conf > 0.3: cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(255, 255,
            # 255), l=15) cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1,
            # thickness=1, offset=3)

            if currentClass == "person" and conf > 0.85:
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(255, 255, 255), l=15)
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

                # find the center of the bounding box
                bb_center_x = (x1 + x2) // 2
                bb_center_y = (y1 + y2) // 2

                # Draw horizontal line passing through the center (frame)
                cv2.line(img, (0, bb_center_y), (img.shape[1], bb_center_y), (255, 0, 0), 2)  # Green color

                # Draw vertical line passing through the center (frame)
                cv2.line(img, (bb_center_x, 0), (bb_center_x, img.shape[0]), (255, 0, 0), 2)  # Blue color

                # Draw a line from the bounding box center to the frame center
                cv2.line(img, (fcenter_x, fcenter_y), (bb_center_x, bb_center_y), (255, 0, 0), 2)  # Blue color

                # Check if the bounding box is inside the rectangle
                if (top_left_x < bb_center_x < bottom_right_x) and (top_left_y < bb_center_y < bottom_right_y):
                    temp = 'Bounding box is inside the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(img, 'GO FORWARD', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                    print('GO FORWARD')

                elif bb_center_x > bottom_right_x:
                    temp = 'Bounding box is to the right of the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(img, "GO LEFT", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                    print('GO LEFT')
                    # UpDownCheck()

                elif bb_center_x < top_left_x:
                    temp = 'Bounding box is to the left of the rectangle'
                    # print(temp)
                    cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(img, 'GO RIGHT', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                    print('GO RIGHT')
                    # UpDownCheck()

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
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

# cv2.destroyAllWindows()

# cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
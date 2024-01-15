"""
Duburi Jetson Detection and Left Right Centering Code
Date: 01/16/2024
Created by Omi, Minoor
"""

import cv2
from jetson_inference import detectNet
import jetson.utils
import imutils
import time
import cvzone

# Load class labels from the labels.txt file
with open("jetson-inference/python/training/detection/ssd/models/duburi/labels.txt", "r") as file:
    class_labels = file.read().splitlines()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Frame Width
cap.set(4, 720)  # Frame Height

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f'Frame Width: {frame_width}')

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Frame Height: {frame_height}')

# Calculate center coordinates of the frame
fcenter_x = frame_width // 2
fcenter_y = frame_height // 2

# Jetson Model
net = detectNet(argv=["--model=jetson-inference/python/training/detection/ssd/models/duburi/ssd-mobilenet.onnx",
                      "--labels=jetson-inference/python/training/detection/ssd/models/duburi/labels.txt",
                      "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.8)

# For FPS Calculation Variable Initialize
prev_frame_time = 0
new_frame_time = 0

while True:
    res, frame = cap.read()
    if not res:
        break
    frame = imutils.resize(frame, width=800)
    img = jetson.utils.cudaFromNumpy(frame)
    detections = net.Detect(img)
    print(f"detections: {detections}")

    # For FPS Counting
    new_frame_time = time.time()

    # Draw a vertical line (green)
    cv2.line(img, (fcenter_x, 0), (fcenter_x, frame_height), (0, 255, 0), 2)

    # Draw a horizontal line (blue)
    cv2.line(img, (0, fcenter_y), (frame_width, fcenter_y), (0, 255, 0), 2)

    # Frame Center threshold box
    rect_width = 250
    rect_height = 250

    # Calculate top-left and bottom-right coordinates of the threshold rectangle
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

    for obj in detections:

        # Bounding Box Coordinates

        left, bottom, right, top = int(obj.Left), int(obj.Bottom), int(obj.Right), int(obj.Top)
        print("----------------- Bounding Box Coordinates --------------------------")
        print(f"left: {left}")
        print(f"bottom: {bottom}")
        print(f"right: {right}")
        print(f"top: {top}")

        # Coordinate Conversion top-left --> x1 y1, bottom right -->  x2 y2;
        x1, y1, x2, y2 = left, top, right, bottom

        # w, h = x2 - x1, y2 - y1

        class_id = int(obj.ClassID)
        class_name = class_labels[class_id]

        """ class name documentation
        0: ''
        1: ''
        """
        label = f"{class_name} ({obj.Confidence:.2f})"

        # find the center of the bounding box
        bb_center_x = (x1 + x2) // 2
        bb_center_y = (y1 + y2) // 2

        # Draw horizontal line passing through the center
        cv2.line(img, (0, bb_center_y), (img.shape[1], bb_center_y), (255, 0, 0), 2)  # Green color

        # Draw vertical line passing through the center
        cv2.line(img, (bb_center_x, 0), (bb_center_x, img.shape[0]), (255, 0, 0), 2)  # Blue color

        # Draw a line from the bounding box center to the frame center
        cv2.line(img, (fcenter_x, fcenter_y), (bb_center_x, bb_center_y), (255, 0, 0), 2)  # Blue color

        # if condition ---------------------------------------

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
        elif bb_center_x < top_left_x:
            temp = 'Bounding box is to the left of the rectangle'
            # print(temp)
            cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(img, 'GO RIGHT', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
            print('GO RIGHT')

        # Don't need this (safety er jonno rekhe disi)
        else:
            temp = 'Bounding box is neither inside nor to the sides of the rectangle'
            print(temp)
            cv2.putText(img, temp, (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
        # cv2.putText(frame, label, (left, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # FPS Measurements
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # --------------------------------- Detection Window ---------------------------------

    # print("Detection FPS: ", fps)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow('Webcam Stream', img)

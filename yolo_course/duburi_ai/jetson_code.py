import cv2
from jetson_inference import detectNet
import jetson.utils
import imutils

# Load class labels from the labels.txt file
with open("jetson-inference/python/training/detection/ssd/models/duburi/labels.txt", "r") as file:
    class_labels = file.read().splitlines()

cap = cv2.VideoCapture(0)
net = detectNet(argv=["--model=jetson-inference/python/training/detection/ssd/models/duburi/ssd-mobilenet.onnx",
                      "--labels=jetson-inference/python/training/detection/ssd/models/duburi/labels.txt",
                      "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.8)

while True:
    res, frame = cap.read()
    if not res:
        break
    frame = imutils.resize(frame, width=800)
    img = jetson.utils.cudaFromNumpy(frame)
    detections = net.Detect(img)

    for obj in detections:
        left, bottom, right, top = int(obj.Left), int(obj.Bottom), int(obj.Right), int(obj.Top)
        class_id = int(obj.ClassID)
        class_name = class_labels[class_id]
        label = f"{class_name} ({obj.Confidence:.2f})"

        cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
        cv2.putText(frame, label, (left, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

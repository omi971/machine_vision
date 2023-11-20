from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # Training.
    results = model.train(
        data="C:\\Users\\Nazmu\\Desktop\\codes\\datasets\\pothole.yaml",
        imgsz=1280,
        epochs=10,
        batch=4,
        name='yolov8n_v8_10e'
        )
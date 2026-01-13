from ultralytics import YOLO
import kagglehub

# Download dataset
dataset_path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")

# Train model
model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset_path}/data.yaml",
    epochs=5,
    imgsz=640,
    batch=16
)

# After training:
# copy runs/detect/train/weights/best.pt → model/pcb_yolo.pt

# predict.py
import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO

# ======================
# USER CONFIG
# ======================
MODEL_PATH = "runs/detect/train3/weights/best.pt"  # trained model path
INPUT_DIR = "sample_images"                        # folder with input images
OUTPUT_DIR = "outputs"                             # folder to save annotated images + CSV
CONF_THRESH = 0.5                                  # min confidence to keep a detection

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Function to calculate severity
def severity_label(conf):
    if conf > 0.85:
        return "high"
    elif conf > 0.65:
        return "medium"
    else:
        return "low"

# Prepare results storage
results_list = []

# Loop over images
for img_file in os.listdir(INPUT_DIR):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, img_file)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Run inference
    pred = model.predict(source=img_path, conf=CONF_THRESH, save=False, verbose=False)[0]

    # Draw boxes + collect data
    for det in pred.boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = det.xyxy[0].tolist()
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Confidence and class
        conf = float(det.conf)
        cls_id = int(det.cls)
        cls_name = model.names[cls_id]

        # Severity
        sev = severity_label(conf)

        # Draw rectangle + label on image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f} ({sev})"
        cv2.putText(img, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Append result
        results_list.append({
            "image": img_file,
            "defect_type": cls_name,
            "confidence": conf,
            "x_center": x_center,
            "y_center": y_center,
            "severity": sev
        })

    # Save annotated image
    out_path = os.path.join(OUTPUT_DIR, img_file)
    cv2.imwrite(out_path, img)

# Save CSV
df = pd.DataFrame(results_list)
csv_path = os.path.join(OUTPUT_DIR, "results.csv")
df.to_csv(csv_path, index=False)

print(f"Inference done! Annotated images + CSV saved to '{OUTPUT_DIR}'")

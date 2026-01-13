import cv2
import json
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "model/pcb_yolo.pt"
IMAGE_DIR = "data/sample_images"
OUTPUT_DIR = "results/visualized"
JSON_PATH = "results/output.json"

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------- Severity ----------------
def estimate_severity(area):
    if area < 500:
        return "Low"
    elif area < 1500:
        return "Medium"
    else:
        return "High"

# ---------------- Load Model ----------------
model = YOLO(MODEL_PATH)

all_results = []

# ---------------- Batch Inference ----------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    results = model(image)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        area = (x2 - x1) * (y2 - y1)

        severity = estimate_severity(area)

        defect = {
            "image": img_name,
            "defect_type": CLASS_NAMES[cls_id],
            "confidence": round(conf, 3),
            "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
            "center_pixel": [cx, cy],
            "severity": severity
        }
        all_results.append(defect)

        # Draw
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f} [{severity}]"
        cv2.putText(image, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)

# ---------------- Save JSON ----------------
with open(JSON_PATH, "w") as f:
    json.dump(all_results, f, indent=4)

print("✅ Batch inspection completed")

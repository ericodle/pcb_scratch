import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from model import create_custom_faster_rcnn
from annotation_parser import get_id_to_label_mapping
import json
import os
from collections import Counter

# === CONFIG ===
TEST_IMAGE_PATH = "./test_imgs/0001_1_eric.JPG"
MODEL_PATH = "./trained_models/faster_rcnn_trained.pth"
SAVE_PATH = "./inference_outputs/"
NUM_CLASSES = 1 + 6  # 1 background + 6 component types
CONFIDENCE_THRESHOLD = 0.9

print("[INFO] Starting inference script...")
os.makedirs(SAVE_PATH, exist_ok=True)

# === LOAD IMAGE ===
transform = T.Compose([T.ToTensor()])
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)
print("[INFO] Image loaded and transformed.")

# === LOAD MODEL ===
model = create_custom_faster_rcnn(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("[INFO] Model loaded and set to evaluation mode.")

# === RUN INFERENCE ===
with torch.no_grad():
    predictions = model(image_tensor)
print("[INFO] Inference complete.")

# === LOAD LABEL MAPPINGS ===
try:
    id_to_label = get_id_to_label_mapping("./train_imgs/")
    print(f"[DEBUG] Label mapping: {id_to_label}")
except Exception as e:
    print(f"[WARNING] Could not load label mappings: {e}")
    id_to_label = {}

# === PARSE PREDICTIONS ===
draw = ImageDraw.Draw(image)
detection_results = []
predicted_counts = Counter()

boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]

for box, label, score in zip(boxes, labels, scores):
    if score >= CONFIDENCE_THRESHOLD:
        box = box.tolist()
        label_id = label.item()
        label_name = id_to_label.get(label_id, f"class_{label_id}")
        predicted_counts[label_name] += 1

        display_text = f"{label_name}: {score:.2f}"
        detection_results.append(display_text)

        # Draw prediction
        text_bbox = draw.textbbox((0, 0), display_text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_origin = (box[0], box[1] - text_h if box[1] - text_h > 0 else box[1] + 5)

        draw.rectangle(box, outline="red", width=2)
        draw.rectangle([text_origin, (text_origin[0] + text_w, text_origin[1] + text_h)], fill="red")
        draw.text(text_origin, display_text, fill="white")

print(f"[INFO] Detected component counts: {dict(predicted_counts)}")

# === LOAD GROUND TRUTH ===
json_path = TEST_IMAGE_PATH.replace(".JPG", ".json").replace(".jpg", ".json")
try:
    with open(json_path, "r") as f:
        annotation = json.load(f)
    print(f"[INFO] Loaded ground truth from {json_path}")
except Exception as e:
    print(f"[ERROR] Failed to load ground truth JSON: {e}")
    annotation = {}

# === COUNT GROUND TRUTH COMPONENTS ===
gt_counts = Counter()
for shape in annotation.get("shapes", []):
    label = shape.get("label", "unknown")
    gt_counts[label] += 1

print(f"[INFO] Ground truth component counts: {dict(gt_counts)}")

# === COMPARE COUNTS ===
all_labels = set(gt_counts) | set(predicted_counts)
discrepancies = []
for label in sorted(all_labels):
    gt = gt_counts[label]
    pred = predicted_counts[label]
    if gt != pred:
        discrepancies.append((label, gt, pred))

# === REPORT ===
report_lines = []

if discrepancies:
    print("[WARNING] Component count mismatches found:")
    report_lines.append("=== Component Count Discrepancies ===")
    for label, gt, pred in discrepancies:
        line = f" - {label}: Ground truth = {gt}, Detected = {pred}"
        print(line)
        report_lines.append(line)
else:
    print("[SUCCESS] All component counts match.")
    report_lines.append("All component counts match.")

# === SAVE ANNOTATED IMAGE ===
img_filename = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", "_pred.jpg").replace(".JPG", "_pred.jpg")
img_save_path = os.path.join(SAVE_PATH, img_filename)
image.save(img_save_path)
print(f"[INFO] Saved annotated image: {img_save_path}")

# === SAVE TEXT REPORT ===
txt_filename = img_filename.replace("_pred.jpg", "_report.txt")
txt_path = os.path.join(SAVE_PATH, txt_filename)

with open(txt_path, "w") as f:
    f.write("=== Detected Objects ===\n")
    for line in detection_results:
        f.write(line + "\n")

    f.write("\n=== Ground Truth Component Counts ===\n")
    for label, count in gt_counts.items():
        f.write(f"{label}: {count}\n")

    f.write("\n=== Predicted Component Counts ===\n")
    for label, count in predicted_counts.items():
        f.write(f"{label}: {count}\n")

    f.write("\n")
    for line in report_lines:
        f.write(line + "\n")

print(f"[INFO] Saved detection and parity report: {txt_path}")

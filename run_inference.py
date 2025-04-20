import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser, get_id_to_label_mapping

# File paths and settings
TEST_IMAGE_PATH = "./test_imgs/0001_1_eric.JPG"
MODEL_PATH = "./trained_models/faster_rcnn_trained.pth"
SAVE_PATH = "./inference_outputs/"
TRAIN_FOLDER = "./train_imgs"  # Folder containing both images and annotations
NUM_CLASSES = 1 + 6  # 1 for background, 6 for component classes
CONFIDENCE_THRESHOLD = 0.9

print("[INFO] Starting inference script...")

# Create output directory if not exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Preprocessing transform
transform = T.Compose([
    T.ToTensor()
])

# Load image
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
print("[INFO] Image loaded and transformed.")

# Load the trained model
model = create_faster_rcnn_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode
print("[INFO] Model loaded and set to evaluation mode.")

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)
print("[INFO] Inference complete.")

# Load label mappings from training data
print("[INFO] Loading label mappings from training data...")
try:
    id_to_label = get_id_to_label_mapping(TRAIN_FOLDER)  # Pass TRAIN_FOLDER here
    print(f"[DEBUG] Label mapping: {id_to_label}")
except Exception as e:
    print(f"[WARNING] Could not load label mappings: {e}")
    print("[INFO] Using fallback label mapping (numeric labels).")
    id_to_label = {}

# Draw predictions and collect detection results
print("[INFO] Drawing predictions on image...")
draw = ImageDraw.Draw(image)
detection_results = []

boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]

for box, label, score in zip(boxes, labels, scores):
    if score >= CONFIDENCE_THRESHOLD:
        box = box.tolist()
        label_id = label.item()
        label_text = id_to_label.get(label_id, f"Class {label_id}")
        display_text = f"{label_text}: {score:.2f}"
        detection_results.append(f"{label_text}: {score:.4f}")

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), display_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Text origin (place above or below the box if out of bounds)
        text_origin = (box[0], box[1] - text_height if box[1] - text_height > 0 else box[1] + 5)

        # Draw bounding box and text background
        draw.rectangle(box, outline="red", width=2)
        draw.rectangle([text_origin, (text_origin[0] + text_width, text_origin[1] + text_height)], fill="red")
        draw.text(text_origin, display_text, fill="white")

# Save predicted image
save_image_name = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", "_predicted.jpg").replace(".JPG", "_predicted.jpg")
output_image_path = os.path.join(SAVE_PATH, save_image_name)
image.save(output_image_path)
print(f"[INFO] Saved prediction image to: {output_image_path}")

# Save detection summary to .txt file
txt_filename = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", "_predictions.txt").replace(".JPG", "_predictions.txt")
txt_path = os.path.join(SAVE_PATH, txt_filename)

with open(txt_path, "w") as f:
    if detection_results:
        f.write("Detected objects and confidence scores:\n")
        for line in detection_results:
            f.write(line + "\n")
    else:
        f.write("No detections above the confidence threshold.\n")

print(f"[INFO] Saved detection summary to: {txt_path}")

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser
import os

# --- Config ---
TEST_IMAGE_PATH = "./test_imgs/0001_1_eric.JPG"
MODEL_PATH = "./trained_models/faster_rcnn_trained.pth"
SAVE_PATH = "./inference_outputs/"
NUM_CLASSES = 1 + 6  # 1 for background, 6 for component classes
CONFIDENCE_THRESHOLD = 0.7
ANNOTATIONS_DIR = "./annotations"

print("[INFO] Starting inference script...")

# Create output directory
os.makedirs(SAVE_PATH, exist_ok=True)

# Preprocessing transform
transform = T.Compose([
    T.ToTensor()
])

# Load image
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
print("[INFO] Image loaded and transformed.")

# Load model
model = create_faster_rcnn_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
print("[INFO] Model loaded and set to evaluation mode.")

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)
print("[INFO] Inference complete.")

# Load label mappings
print("[INFO] Loading label mappings from training data...")
try:
    dummy_dataset = CustomAnnotationParser(folder_path=ANNOTATIONS_DIR)
    if dummy_dataset.label_to_id:
        id_to_label = {v: k for k, v in dummy_dataset.label_to_id.items()}
        print(f"[DEBUG] Label mapping: {id_to_label}")
    else:
        raise ValueError("No labels found in training data.")
except Exception as e:
    print(f"[WARNING] Could not load label mappings: {e}")
    print("[INFO] Using fallback label mapping (numeric labels).")
    id_to_label = {}

# Draw predictions
print("[INFO] Drawing predictions on image...")
draw = ImageDraw.Draw(image)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 16)  # You can adjust font size here if needed
except:
    print("[WARNING] 'arial.ttf' not found, using default font.")
    font = ImageFont.load_default()

boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]

for box, label, score in zip(boxes, labels, scores):
    if score >= CONFIDENCE_THRESHOLD:
        box = box.tolist()
        label_id = label.item()
        label_text = id_to_label.get(label_id, f"Class {label_id}")
        display_text = f"{label_text}: {score:.2f}"

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Text origin (place above or below the box if out of bounds)
        text_origin = (box[0], box[1] - text_height if box[1] - text_height > 0 else box[1] + 5)

        # Draw bounding box and text background
        draw.rectangle(box, outline="red", width=2)
        draw.rectangle([text_origin, (text_origin[0] + text_width, text_origin[1] + text_height)], fill="red")
        draw.text(text_origin, display_text, fill="white", font=font)

# Save result
save_name = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", "_predicted.jpg")
output_path = os.path.join(SAVE_PATH, save_name)
image.save(output_path)
print(f"[INFO] Saved prediction to: {output_path}")

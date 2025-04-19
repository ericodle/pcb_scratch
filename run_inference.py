import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser
import os

# --- Config ---
TEST_IMAGE_PATH = "./test_imgs/0001_1_eric.JPG"  # Replace with actual filename
MODEL_PATH = "./trained_models/faster_rcnn_model.pth"  # Replace with your trained model path
SAVE_PATH = "./inference_outputs/"
NUM_CLASSES = 1 + 10  # Update to match your training
CONFIDENCE_THRESHOLD = 0.5

# Create output directory
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Preprocessing transform ---
transform = T.Compose([
    T.ToTensor()
])

# --- Load image ---
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# --- Load model ---
model = create_faster_rcnn_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- Run inference ---
with torch.no_grad():
    predictions = model(image_tensor)

# --- Draw predictions ---
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# If you used label mappings during training:
dummy_dataset = CustomAnnotationParser(folder_path="./pcb_scratch/")
id_to_label = {v: k for k, v in dummy_dataset.label_to_id.items()}

boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]

for box, label, score in zip(boxes, labels, scores):
    if score >= CONFIDENCE_THRESHOLD:
        box = box.tolist()
        label_name = id_to_label.get(label.item(), str(label.item()))
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), f"{label_name}: {score:.2f}", fill="yellow", font=font)

# --- Save result ---
save_name = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", "_predicted.jpg")
output_path = os.path.join(SAVE_PATH, save_name)
image.save(output_path)
print(f"Saved prediction to: {output_path}")

import os
import time
import torch
import logging
import matplotlib.pyplot as plt
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

# Setup logger
logging.basicConfig(filename='logs/training.log', level=logging.INFO)
logger = logging.getLogger()

# File paths and settings
FOLDER_PATH = "train_imgs"
NUM_CLASSES = 1 + 6  # 1 for background + 6 component classes

def transform(image, target):
    image = F.to_tensor(image)
    return image, target

def create_optimizer(model, learning_rate=0.001):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

# -----------------------------
# IoU CALCULATION UTILITIES
# -----------------------------
def compute_iou(box1, box2):
    """
    box1: list or tensor [x1, y1, x2, y2]
    box2: list or tensor [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union

def evaluate_iou(model, dataset, device, num_samples=10):
    model.eval()
    ious = []

    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            image, target = dataset[idx]
            image = image.to(device)
            prediction = model([image])[0]

            pred_boxes = prediction['boxes'].cpu()
            gt_boxes = target['boxes'].cpu()

            for gt_box in gt_boxes:
                max_iou = 0
                for pred_box in pred_boxes:
                    iou = compute_iou(gt_box.tolist(), pred_box.tolist())
                    max_iou = max(max_iou, iou)
                ious.append(max_iou)

    model.train()
    return sum(ious) / len(ious) if ious else 0.0

# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model_with_loss_tracking(model, data_loader, device, num_epochs=15):
    model.to(device)
    model.train()
    optimizer = create_optimizer(model)

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Losses: {loss_dict}")

            loss_classifier_values.append(loss_dict['loss_classifier'].item())
            loss_box_reg_values.append(loss_dict['loss_box_reg'].item())
            loss_objectness_values.append(loss_dict['loss_objectness'].item())
            loss_rpn_box_reg_values.append(loss_dict['loss_rpn_box_reg'].item())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Evaluate IoU after each epoch
        mean_iou = evaluate_iou(model, dataset, device)
        iou_values.append(mean_iou)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Mean IoU: {mean_iou:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Mean IoU: {mean_iou:.4f}")

    # Plot all loss components and IoU
    plot_loss("loss_classifier", loss_classifier_values)
    plot_loss("loss_box_reg", loss_box_reg_values)
    plot_loss("loss_objectness", loss_objectness_values)
    plot_loss("loss_rpn_box_reg", loss_rpn_box_reg_values)
    plot_loss("mean_iou", iou_values)

    # Save the trained model
    os.makedirs('trained_models', exist_ok=True)
    torch.save(model.state_dict(), 'trained_models/faster_rcnn_trained.pth')
    print("Model saved to 'trained_models/faster_rcnn_trained.pth'")
    logger.info("Model saved to 'trained_models/faster_rcnn_trained.pth'")

# -----------------------------
# LOSS / METRIC PLOTTING
# -----------------------------
def plot_loss(loss_name, loss_values):
    os.makedirs("train_plots", exist_ok=True)
    plt.figure()
    plt.plot(loss_values)
    plt.title(f'{loss_name} over epochs')
    plt.xlabel('Epoch')
    plt.ylabel(loss_name)
    plt.savefig(f"train_plots/{loss_name}_plot.png")
    plt.close()

# -----------------------------
# DATASET AND TRAINING LAUNCH
# -----------------------------
dataset = CustomAnnotationParser(folder_path=FOLDER_PATH, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_faster_rcnn_model(NUM_CLASSES)

# Loss tracking
loss_classifier_values = []
loss_box_reg_values = []
loss_objectness_values = []
loss_rpn_box_reg_values = []
iou_values = []

train_model_with_loss_tracking(model, data_loader, device)

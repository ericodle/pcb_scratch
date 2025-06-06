import os
import time
import torch
import logging
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from model import create_custom_faster_rcnn
from annotation_parser import CustomAnnotationParser
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

# Setup logger
logging.basicConfig(filename='logs/training.log', level=logging.INFO)
logger = logging.getLogger()

# File paths and settings
FOLDER_PATH = "train_imgs"
NUM_CLASSES = 1 + 6  # 1 for background + 6 component classes

# -----------------------------
# DATA AUGMENTATION WITH ALBUMENTATIONS
# -----------------------------
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.0625), rotate=(-10, 10), p=0.5),
    A.RandomSizedBBoxSafeCrop(height=512, width=512, p=0.5),
    A.Blur(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

USE_ALBUMENTATIONS = False

def transform(image, target):
    if not USE_ALBUMENTATIONS:
        # Just convert the image and target to tensors and return
        image = F.to_tensor(image)
        target['boxes'] = target['boxes'].float()
        target['labels'] = target['labels'].long()
        return image, target

    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Convert boxes from Tensor to list format
    bboxes = target['boxes'].tolist()
    labels = target['labels'].tolist()

    transformed = albumentations_transform(image=image_np, bboxes=bboxes, labels=labels)

    image = transformed['image']
    target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
    target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

    return image, target

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
    area2 = (box2[2] - box2[0]) * (box2[3] - box1[1])
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

import torch.optim as optim

def create_optimizer(model, lr=1e-4):
    """
    Create an optimizer for Faster R-CNN model
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Create the Adam optimizer with the specified learning rate
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    
    return optimizer
# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model_with_loss_tracking(model, data_loader, device, num_epochs=20):
    model.to(device)
    print(f"Using device: {device}")

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
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_custom_faster_rcnn(NUM_CLASSES)

# Loss tracking
loss_classifier_values = []
loss_box_reg_values = []
loss_objectness_values = []
loss_rpn_box_reg_values = []
iou_values = []

train_model_with_loss_tracking(model, data_loader, device)

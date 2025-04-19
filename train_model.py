# train_model.py

import torch
from train import train_model
from model import create_faster_rcnn_model
from torch.utils.data import DataLoader
from annotation_parser import CustomAnnotationParser
from utils.transforms import get_transform

# Set paths for your dataset and annotations
ANNOTATION_FILE = "/path/to/annotations.json"
IMAGES_DIR = "/path/to/images"

# Number of classes (including background)
num_classes = 2  # Modify according to your dataset, including background as class 0

# Initialize the dataset and DataLoader
dataset = CustomAnnotationParser(ANNOTATION_FILE, IMAGES_DIR, transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Create the model
model = create_faster_rcnn_model(num_classes)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Start training
train_model(model, data_loader, device, num_epochs=10)

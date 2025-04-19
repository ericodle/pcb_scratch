# train_model.py

import torch
from train import train_model
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser
from transforms import get_transform
from torch.utils.data import DataLoader

ANNOTATION_FILE = "/home/eo/pcb_scratch/0009_1_yuka.json"
IMAGES_DIR = "/home/eo/pcb_scratch/"

# Count your labels + 1 for background
NUM_CLASSES = 1 + 10  # Example if you have 10 classes

dataset = CustomAnnotationParser(ANNOTATION_FILE, IMAGES_DIR, transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_faster_rcnn_model(NUM_CLASSES)

train_model(model, data_loader, device, num_epochs=10)

import os
import time
import torch
import logging
import matplotlib.pyplot as plt
from model import create_faster_rcnn_model
from annotation_parser import CustomAnnotationParser
from transforms import get_transform
from torch.utils.data import DataLoader

# Setup logger
logging.basicConfig(filename='training.log', level=logging.INFO)
logger = logging.getLogger()

# File paths and settings
FOLDER_PATH = "train_imgs"   # Folder containing both images and annotations
IMAGES_DIR = os.path.join(FOLDER_PATH, "images")  # Assuming images are stored in a subfolder 'images'
ANNOTATIONS_DIR = os.path.join(FOLDER_PATH, "annotations")  # Assuming annotations are in a subfolder 'annotations'
NUM_CLASSES = 1 + 10  # Example with 10 classes

# Data setup
dataset = CustomAnnotationParser(folder_path=FOLDER_PATH, transform=get_transform(train=True))

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_faster_rcnn_model(NUM_CLASSES)

# Initialize loss lists for plotting
loss_classifier_values = []
loss_box_reg_values = []
loss_objectness_values = []
loss_rpn_box_reg_values = []

def create_optimizer(model, learning_rate=0.005):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

def train_model_with_loss_tracking(model, data_loader, device, num_epochs=30):
    model.to(device)
    model.train()

    # Initialize the optimizer here
    optimizer = create_optimizer(model)  # Create optimizer using your function

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch+1}/{num_epochs}...")  # Print info for the current epoch

        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Log detailed losses
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Losses: {loss_dict}")

            # Track individual losses for plotting
            loss_classifier_values.append(loss_dict['loss_classifier'].item())
            loss_box_reg_values.append(loss_dict['loss_box_reg'].item())
            loss_objectness_values.append(loss_dict['loss_objectness'].item())
            loss_rpn_box_reg_values.append(loss_dict['loss_rpn_box_reg'].item())

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero gradients
            losses.backward()      # Backpropagation
            optimizer.step()       # Update model parameters

            total_loss += losses.item()

        # Average loss for this epoch
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")  # Print avg loss to the console

    # Plot loss components after training
    plot_loss("loss_classifier", loss_classifier_values)
    plot_loss("loss_box_reg", loss_box_reg_values)
    plot_loss("loss_objectness", loss_objectness_values)
    plot_loss("loss_rpn_box_reg", loss_rpn_box_reg_values)

def plot_loss(loss_name, loss_values):
    plt.figure()
    plt.plot(loss_values)
    plt.title(f'train_plots/{loss_name} over epochs')
    plt.xlabel('train_plots/Iteration')
    plt.ylabel(f'train_plots/{loss_name} value')
    plt.savefig(f"train_plots/{loss_name}_plot.png")  # Save the plot as a PNG file
    plt.close()


# Start training with logging and plotting
train_model_with_loss_tracking(model, data_loader, device)

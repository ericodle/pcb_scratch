import torch
import matplotlib.pyplot as plt
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

# Move the model to the correct device
model.to(device)

# List to store loss values for plotting and detailed tracking
losses = []
detailed_losses = []  # List to track detailed loss components

# Function to save the plot
def save_loss_plot(losses, filename="loss_plot.png"):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig(filename)
    print(f"Loss plot saved as {filename}")
    plt.close()

# Function to save the model
def save_model(model, filename="faster_rcnn_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Modify your train_model function to store the loss for each epoch and detailed loss components
def train_model_with_loss_tracking(model, data_loader, device, num_epochs=10):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        epoch_loss = 0  # Track loss for each epoch
        for images, targets in data_loader:
            # Move the images and targets to the correct device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())  # Total loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()

            # Record detailed loss components
            detailed_loss = {
                "loss_classifier": loss_dict["loss_classifier"].item(),
                "loss_box_reg": loss_dict["loss_box_reg"].item(),
                "loss_objectness": loss_dict["loss_objectness"].item(),
                "loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"].item(),
            }
            detailed_losses.append(detailed_loss)
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)  # Append the average loss of this epoch to the list
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    save_loss_plot(losses)  # Save the loss plot
    save_model(model)  # Save the model

# Run the training with loss tracking
train_model_with_loss_tracking(model, data_loader, device, num_epochs=10)

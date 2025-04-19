# train.py

import time
import torch
from torchvision import datasets, transforms
from engine import train_one_epoch, evaluate
import utils

def create_optimizer(model, learning_rate=0.005):
    """
    Creates the optimizer for the model.
    
    Args:
        model (torch.nn.Module): The model to be optimized.
        learning_rate (float): The learning rate for the optimizer.
    
    Returns:
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

def create_lr_scheduler(optimizer, step_size=3, gamma=0.1):
    """
    Creates a learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        step_size (int): The step size for the scheduler.
        gamma (float): The learning rate decay factor.
    
    Returns:
        lr_scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler.
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return lr_scheduler

def train_model(model, data_loader, device, num_epochs=10):
    """
    Train the Faster R-CNN model.
    
    Args:
        model (FasterRCNN): The model to train.
        data_loader (DataLoader): The data loader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run the model on.
        num_epochs (int): The number of epochs to train for.
    """
    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer)

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        model.train()
        start = time.time()
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluation phase
        model.eval()
        evaluate(model, data_loader, device=device)

        print(f"Epoch {epoch} completed in {time.time() - start:.2f} seconds.")

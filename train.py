# train.py

import time
import torch
from torchvision import datasets, transforms
import utils

def create_optimizer(model, learning_rate=0.005):

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

def create_lr_scheduler(optimizer, step_size=3, gamma=0.1):

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return lr_scheduler

def train_model(model, data_loader, device, num_epochs=10):

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

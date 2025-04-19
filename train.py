import time
import torch
from collections import defaultdict
import numpy as np


def create_optimizer(model, learning_rate=0.005):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer


def create_lr_scheduler(optimizer, step_size=3, gamma=0.1):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return lr_scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = defaultdict(list)

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]  # Move images to the device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the device

        loss_dict = model(images, targets)  # Forward pass
        losses = sum(loss for loss in loss_dict.values())  # Sum all losses

        optimizer.zero_grad()  # Zero the gradients
        losses.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        for k, v in loss_dict.items():
            metric_logger[k].append(v.item())  # Track losses

        if i % print_freq == 0:
            print(f"[Epoch {epoch} | Iter {i}] Loss: {losses.item():.4f}")

    print(f"\n[Epoch {epoch}] Training Summary:")
    for k, v in metric_logger.items():
        print(f"  {k}: {np.mean(v):.4f}")


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    print("\nRunning evaluation...")

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]  # Move images to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

        outputs = model(images)  # Model inference

        print(f"Batch {i + 1}:")
        for j, output in enumerate(outputs):
            num_preds = len(output["boxes"])
            print(f"  Image {j}: {num_preds} predictions")


def train_model(model, data_loader, device, num_epochs=10):
    model.to(device)  # Ensure the model is on the correct device (CPU/GPU)

    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer)

    for epoch in range(num_epochs):
        start = time.time()

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()  # Adjust learning rate
        evaluate(model, data_loader, device)  # Evaluate the model

        print(f"[Epoch {epoch}] Completed in {time.time() - start:.2f} seconds.\n")

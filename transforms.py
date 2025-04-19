# transforms.py

import torchvision.transforms as T

def get_transform(train=True):
    """
    Returns the transformations to be applied to the images during training or testing.
    """
    transforms = []
    transforms.append(T.ToTensor())  # Convert the image to a PyTorch tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Random horizontal flip with 50% probability
    return T.Compose(transforms)  # Combine the transformations into a single callable

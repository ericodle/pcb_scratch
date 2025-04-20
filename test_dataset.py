import os
from annotation_parser import CustomAnnotationParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            width = image.shape[-1]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)



# Update this path to your folder containing images and JSON annotations
FOLDER_PATH = "train_imgs"  # Folder containing both images and annotations
OUTPUT_DIR = "reconstructed_images"  # Folder to save the output images

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def visualize_sample(image, target, output_filename):
    # Convert from [C, H, W] (Tensor) to [H, W, C] (NumPy)
    image = image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in target["boxes"]:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Save the image in the specified output folder
    plt.savefig(f"{OUTPUT_DIR}/{output_filename}")
    plt.close(fig)

def main():
    # Initialize the dataset and dataloader with the transform
    transform = get_transform(train=False)  # Use transform for testing (False for no training augmentations)
    
    # Initialize the dataset for the folder with multiple images and annotations
    dataset = CustomAnnotationParser(FOLDER_PATH, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")  # Debugging the length of the dataset
    
    if len(dataset) == 0:
        print("No valid data found in dataset.")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Iterate through all samples in the dataset
    for idx, (images, targets) in enumerate(dataloader):
        img = images[0]
        tgt = targets[0]
        print(f"Image size: {img.size()}, Boxes: {tgt['boxes'].shape}, Labels: {tgt['labels']}")

        # Create a unique filename for each image
        output_filename = f"image_{idx + 1}.png"
        
        # Visualize and save the sample
        visualize_sample(img, tgt, output_filename)

if __name__ == "__main__":
    main()

from annotation_parser import CustomAnnotationParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from transforms import get_transform  # Import the transform function

# Update these paths to match your project structure
FOLDER_PATH = "test_imgs"  # Folder containing both images and annotations

def visualize_sample(image, target):
    # Convert from [C, H, W] (Tensor) to [H, W, C] (NumPy)
    image = image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in target["boxes"]:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("output_image.png")
    plt.close(fig)

def main():
    # Initialize the dataset and dataloader with the transform
    transform = get_transform(train=False)  # Use transform for testing (False for no training augmentations)
    
    # Pass folder_path instead of single annotation file
    dataset = CustomAnnotationParser(FOLDER_PATH, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")  # Debugging the length of the dataset
    
    if len(dataset) == 0:
        print("No valid data found in dataset.")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Iterate through one sample in the dataset
    for images, targets in dataloader:
        img = images[0]
        tgt = targets[0]
        print(f"Image size: {img.size()}, Boxes: {tgt['boxes'].shape}, Labels: {tgt['labels']}")
        visualize_sample(img, tgt)
        break  # Test one sample only

if __name__ == "__main__":
    main()

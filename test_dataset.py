# test_dataset.py

from annotation_parser import CustomAnnotationParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import numpy as np

# Update these paths to match your project structure
ANNOTATION_FILE = "/home/eo/pcb_scratch/0009_1_yuka.json"
IMAGES_DIR = "/home/eo/pcb_scratch/"

def visualize_sample(image, target):
    image = np.array(image)  # Convert PIL image to NumPy array (HWC format)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in target["boxes"]:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("output_image.png")  # Save the image instead of showing it
    plt.close(fig)  # Close the figure to avoid memory issues

def main():
    # Initialize the dataset and dataloader
    dataset = CustomAnnotationParser(ANNOTATION_FILE, IMAGES_DIR)
    
    print(f"Dataset length: {len(dataset)}")  # Debugging the length of the dataset
    
    if len(dataset) == 0:
        print("No valid data found in dataset.")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Iterate through one sample in the dataset
    for images, targets in dataloader:
        img = images[0]
        tgt = targets[0]
        print(f"Image size: {img.size}, Boxes: {tgt['boxes'].shape}, Labels: {tgt['labels']}")
        visualize_sample(img, tgt)
        break  # Test one sample only

if __name__ == "__main__":
    main()

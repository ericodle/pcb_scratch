from annotation_parser import CustomAnnotationParser  # Import your dataset class
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define the transformation (for training only)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor
    transforms.RandomHorizontalFlip()  # Data augmentation
])

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

    plt.show()

def main():
    # Initialize the dataset and dataloader with the transform
    dataset = CustomAnnotationParser(ANNOTATION_FILE, IMAGES_DIR, transform=transform)
    
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

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomAnnotationParser(Dataset):
    def __init__(self, annotation_file, images_dir):
        self.annotation_file = annotation_file
        self.images_dir = images_dir

        # Load annotations
        with open(self.annotation_file) as f:
            self.annotations = json.load(f)["data"]

        # Load image names
        self.image_names = [image["filename"] for image in self.annotations["images"]]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the image
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Get the annotations for the image
        image_annotations = self.get_annotations_for_image(image_name)

        # Convert bounding boxes and labels
        boxes = []
        labels = []
        for annotation in image_annotations:
            x_min, y_min, x_max, y_max = annotation["bbox"]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation["category_id"])

        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Return image and target (bounding boxes, labels)
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target

    def get_annotations_for_image(self, image_name):
        # Filter annotations for the given image
        return [ann for ann in self.annotations["annotations"] if ann["image_id"] == image_name]

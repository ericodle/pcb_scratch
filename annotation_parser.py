import json
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image

class CustomAnnotationParser:
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path) as f:
            annotations_data = json.load(f)

        self.images_dir = Path(images_dir)
        self.transform = transform  # 👈 New: store the transform

        # Extract image filename from JSON metadata
        self.image_filename = annotations_data.get("imagePath", "")
        
        # Extract shapes (objects)
        self.shapes = annotations_data.get("shapes", [])

        # Creating a mapping of labels to indices
        self.label_to_id = {shape["label"]: idx + 1 for idx, shape in enumerate(self.shapes)}

        # Convert polygons to bounding boxes
        self.annotations = self.convert_polygons_to_boxes(self.shapes)

    def convert_polygons_to_boxes(self, shapes):
        boxes = []
        labels = []

        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            
            x_min = min(x_points)
            y_min = min(y_points)
            x_max = max(x_points)
            y_max = max(y_points)
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_to_id.get(label, 0))

        return boxes, labels

    def __len__(self):
        return 1  # One image per JSON

    def __getitem__(self, idx):
        image = Image.open(self.images_dir / self.image_filename).convert("RGB")

        boxes = torch.tensor(self.annotations[0], dtype=torch.float32)
        labels = torch.tensor(self.annotations[1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }

        # 👇 Apply transform if provided (only to image)
        if self.transform:
            image = self.transform(image)

        return image, target

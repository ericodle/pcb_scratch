# annotation_parser.py

import json
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image

class CustomAnnotationParser:
    def __init__(self, json_path, images_dir):
        with open(json_path) as f:
            annotations_data = json.load(f)

        self.images_dir = Path(images_dir)
        
        # Extract image filename from JSON metadata (assuming the image filename is fixed)
        self.image_filename = annotations_data.get("imagePath", "")
        
        # Extract shapes (objects)
        self.shapes = annotations_data.get("shapes", [])

        # Creating a mapping of labels to indices
        self.label_to_id = {shape["label"]: idx + 1 for idx, shape in enumerate(self.shapes)}

        # Convert polygons to bounding boxes
        self.annotations = self.convert_polygons_to_boxes(self.shapes)

    def convert_polygons_to_boxes(self, shapes):
        """
        Converts polygons to bounding boxes (x_min, y_min, x_max, y_max).
        """
        boxes = []
        labels = []

        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            
            # Get min/max of the points to form a bounding box
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            
            x_min = min(x_points)
            y_min = min(y_points)
            x_max = max(x_points)
            y_max = max(y_points)
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_to_id.get(label, 0))  # Default label 0 for unknown labels

        return boxes, labels

    def __len__(self):
        return 1  # Assuming one image per JSON for simplicity

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.images_dir / self.image_filename).convert("RGB")

        # Convert the boxes to tensors
        boxes = torch.tensor(self.annotations[0], dtype=torch.float32)
        labels = torch.tensor(self.annotations[1], dtype=torch.int64)

        # Return the image and target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }

        return image, target

# annotation_parser.py

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomAnnotationParser(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path) as f:
            annotations_data = json.load(f)

        self.images_dir = Path(images_dir)
        self.transform = transform

        self.image_filename = annotations_data.get("imagePath", "")
        self.shapes = annotations_data.get("shapes", [])

        # Map label to ID
        unique_labels = list(set([s["label"] for s in self.shapes]))
        self.label_to_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}

        self.boxes, self.labels = self.convert_polygons_to_boxes(self.shapes)

    def convert_polygons_to_boxes(self, shapes):
        boxes = []
        labels = []
        for shape in shapes:
            points = shape["points"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_to_id[shape["label"]])
        return boxes, labels

    def __len__(self):
        return 1  # One image per JSON

    def __getitem__(self, idx):
        image = Image.open(self.images_dir / self.image_filename).convert("RGB")

        boxes = torch.tensor(self.boxes, dtype=torch.float32)
        labels = torch.tensor(self.labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

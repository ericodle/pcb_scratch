import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomAnnotationParser(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        # List all the JSON files in the folder
        self.annotation_files = list(self.folder_path.glob('*.json'))

        # Prepare a list to hold image and annotation paths
        self.image_paths = []
        self.annotation_data = []
        
        for ann_file in self.annotation_files:
            with open(ann_file) as f:
                annotations_data = json.load(f)

            image_filename = annotations_data.get("imagePath", "")
            if not image_filename:
                continue

            # Get the corresponding image path
            image_path = self.folder_path / image_filename
            if image_path.exists():
                self.image_paths.append(image_path)
                self.annotation_data.append(annotations_data)

        # Initialize label mappings
        self.label_to_id = self.create_label_mapping()

    def create_label_mapping(self):
        # Create a unique mapping from labels to IDs
        all_labels = set()
        for ann in self.annotation_data:
            for shape in ann.get("shapes", []):
                all_labels.add(shape["label"])
        
        return {label: idx + 1 for idx, label in enumerate(all_labels)}

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
        return len(self.image_paths)  # Return the number of image files

    def __getitem__(self, idx):
        # Get the image path and corresponding annotation
        image_path = self.image_paths[idx]
        annotation = self.annotation_data[idx]

        image = Image.open(image_path).convert("RGB")

        boxes, labels = self.convert_polygons_to_boxes(annotation.get("shapes", []))
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

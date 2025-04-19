# pcb_dataset.py

from torch.utils.data import Dataset
from annotation_parser import CustomCocoParser
from preprocess import get_transform


class PCBDataset(Dataset):
    def __init__(self, annotation_json, images_dir, train=True):
        self.parser = CustomCocoParser(annotation_json, images_dir)
        self.transform = get_transform(train)

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, idx):
        image, target = self.parser.get_item(idx)
        image, target = self.transform(image, target)
        return image, target

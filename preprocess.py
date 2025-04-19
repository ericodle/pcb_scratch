
import torch
import torchvision.transforms.functional as F
import random


def get_transform(train: bool = True):
    def transform(image, target):
        image = F.to_tensor(image)

        if train:
            # Example: random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)

                bbox = target["boxes"]
                bbox[:, [0, 2]] = image.shape[2] - bbox[:, [2, 0]]  # flip x1 and x2
                target["boxes"] = bbox

        return image, target

    return transform

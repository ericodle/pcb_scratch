import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18


def create_pcb_backbone():
    # Use ResNet-18 with pretrained weights
    backbone = resnet18(weights="DEFAULT")

    # Wrap backbone with FPN
    fpn_backbone = BackboneWithFPN(
        backbone,
        return_layers={
            "layer1": "0",
            "layer2": "1",
            "layer3": "2",
            "layer4": "3"
        },
        in_channels_list=[64, 128, 256, 512],  # Channels for ResNet-18
        out_channels=256  # FPN output size
    )

    return fpn_backbone


def create_custom_faster_rcnn(num_classes):
    # Use the lightweight backbone
    backbone = create_pcb_backbone()

    # Build the model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes
    )

    return model

# model.py

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_faster_rcnn_model(num_classes):

    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the input features of the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the classifier to match the number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


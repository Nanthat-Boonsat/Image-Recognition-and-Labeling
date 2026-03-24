import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(backbone: str, num_classes: int = 2):
    """Create Faster R-CNN and replace its classifier head for this task."""
    # Load a pretrained detector backbone for transfer learning.
    if backbone == 'fasterrcnn_resnet50_fpn':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    else:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace COCO head with a dataset-specific classification head.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
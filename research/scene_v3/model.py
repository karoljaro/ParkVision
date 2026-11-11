"""Local-only Mask R-CNN initialized from the existing COCO detector."""

from contextlib import contextmanager

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from research.occupancy.models import DETECTOR_SHA256
from .data import file_hash

ARCHITECTURE = "maskrcnn_resnet50_fpn_v2_vehicle_instance_v3"
RECIPE = {"rgb": True, "min_size": 640, "max_size": 960, "mask_threshold": 0.5}


def build_model(detector_path=None):
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=RECIPE["min_size"],
        max_size=RECIPE["max_size"],
        box_detections_per_img=100,
    )
    if detector_path is not None:
        if file_hash(detector_path) != DETECTOR_SHA256:
            raise ValueError("Unexpected frozen detector initialization")
        state = torch.load(detector_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("roi_heads.mask_") for key in missing):
            raise ValueError("Detector/Mask R-CNN initialization mismatch")
    old = model.roi_heads.box_predictor
    predictor = FastRCNNPredictor(old.cls_score.in_features, 2)
    # Explicit taxonomy: a generic vehicle instance, not a parking state.
    with torch.no_grad():
        predictor.cls_score.weight[0].copy_(old.cls_score.weight[0])
        predictor.cls_score.bias[0].copy_(old.cls_score.bias[0])
        vehicle = [2, 3, 4, 6, 8]  # COCO bicycle, car, motorcycle, bus, truck
        predictor.cls_score.weight[1].copy_(old.cls_score.weight[vehicle].mean(0))
        predictor.cls_score.bias[1].copy_(old.cls_score.bias[vehicle].mean(0))
        weights = old.bbox_pred.weight.reshape(91, 4, -1)
        bias = old.bbox_pred.bias.reshape(91, 4)
        predictor.bbox_pred.weight[:4].copy_(weights[0])
        predictor.bbox_pred.bias[:4].copy_(bias[0])
        predictor.bbox_pred.weight[4:].copy_(weights[vehicle].mean(0))
        predictor.bbox_pred.bias[4:].copy_(bias[vehicle].mean(0))
    model.roi_heads.box_predictor = predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2
    )
    # Keep pretrained low-level features, adapt FPN and later ResNet stages.
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(name.startswith(("layer3", "layer4")))
    return model


@contextmanager
def mask_supervision(model, enabled):
    """BBox-only batches contribute no invented mask targets or mask loss."""
    head = model.roi_heads.mask_predictor
    if not enabled:
        model.roi_heads.mask_predictor = None
    try:
        yield
    finally:
        model.roi_heads.mask_predictor = head


def load_checkpoint(path, expected_sha256, device):
    if file_hash(path) != expected_sha256:
        raise ValueError("Scene checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("architecture") != ARCHITECTURE
        or checkpoint.get("recipe") != RECIPE
        or checkpoint.get("classes") != ["background", "vehicle_instance"]
    ):
        raise ValueError("Scene checkpoint contract mismatch")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device), checkpoint

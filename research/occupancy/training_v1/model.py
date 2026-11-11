"""ConvNeXt with explicit target footprint and a typed abstaining contract."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
import torch
from torch import nn
from torchvision.models import convnext_tiny

from .data import file_hash

ARCHITECTURE = "convnext_tiny_imagenet1k_v1_rgb_roi4"
State = Literal["free", "occupied", "unknown"]


@dataclass(frozen=True, slots=True)
class SlotPrediction:
    """View-local estimate; cross-camera physical IDs need operator mapping."""

    slot_projection_id: str
    state: State
    occupied_score: float
    reason: str | None


def decide(
    slot_projection_id: str,
    score: float,
    *,
    view_confirmed=False,
    visibility: Literal["visible", "occluded", "unassessed"] = "unassessed",
    legal_slot=False,
    free_max=0.10,
    occupied_min=0.90
) -> SlotPrediction:
    if not slot_projection_id.strip():
        raise ValueError("Missing target slot")
    if not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid occupied score")
    if not 0 <= free_max < occupied_min <= 1:
        raise ValueError("Overlapping decision thresholds")
    if visibility not in {"visible", "occluded", "unassessed"}:
        raise ValueError("Invalid visibility")
    reason = None
    state: State = "unknown"
    if not legal_slot:
        reason = "not_confirmed_parking_slot"
    elif not view_confirmed:
        reason = "view_unconfirmed"
    elif visibility != "visible":
        reason = "visibility_" + visibility
    elif score <= free_max:
        state = "free"
    elif score >= occupied_min:
        state = "occupied"
    else:
        reason = "score_abstention"
    return SlotPrediction(slot_projection_id, state, score, reason)


def build_model(pretrained=None, expected_sha256=None):
    model = convnext_tiny(weights=None)
    if pretrained is not None:
        if not expected_sha256 or file_hash(pretrained) != expected_sha256:
            raise ValueError("Pretrained checkpoint hash mismatch")
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    old = model.features[0][0]
    stem = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        stem.weight[:, :3].copy_(old.weight)
        stem.weight[:, 3:].zero_()
        stem.bias.copy_(old.bias)
    model.features[0][0] = stem
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model


def tensor_input(array: np.ndarray, augment=False) -> torch.Tensor:
    if array.shape != (224, 224, 4) or array.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB+ROI array")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255
    if augment:
        # Joint geometry; RGB-only illumination changes. Never random-crop ROI.
        if torch.rand(()) < 0.5:
            tensor = tensor.flip(2)
        gain = 0.80 + 0.40 * torch.rand(())
        offset = -0.06 + 0.12 * torch.rand(())
        tensor[:3] = (tensor[:3] * gain + offset).clamp(0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    tensor[:3] = (tensor[:3] - mean) / std
    return tensor

"""Guard the abstention contract and trainable fourth input channel."""

import numpy as np
import pytest
import torch

from .model import build_model, decide, tensor_input


@pytest.mark.parametrize(
    "score,state",
    [
        (0, "free"),
        (0.1, "free"),
        (0.11, "unknown"),
        (0.5, "unknown"),
        (0.9, "occupied"),
        (1, "occupied"),
    ],
)
def test_fixed_thresholds_with_verified_visibility(score, state):
    result = decide(
        "camera:slot", score, legal_slot=True, view_confirmed=True, visibility="visible"
    )
    assert result.state == state


def test_empty_source_does_not_prove_visibility_or_legal_slot():
    assert decide("T-01", 0).state == "unknown"
    assert decide("C", 0, legal_slot=True, view_confirmed=True).state == "unknown"
    assert (
        decide(
            "C", 1, legal_slot=True, view_confirmed=True, visibility="occluded"
        ).state
        == "unknown"
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError):
        decide("C", score)


def test_mask_is_not_rgb_normalized_or_photometrically_augmented():
    array = np.full((224, 224, 4), 127, np.uint8)
    array[:, :, 3] = 0
    array[20:70, 30:90, 3] = 255
    tensor = tensor_input(array, augment=True)
    assert tensor.shape == (4, 224, 224)
    assert set(tensor[3].unique().tolist()) == {0, 1}
    assert tensor[3].sum() == 50 * 60


def test_real_network_backpropagates_into_mask_channel():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model()
    assert model.features[0][0].in_channels == 4
    assert model.features[0][0].weight[:, 3].count_nonzero() == 0
    logits = model(torch.randn(2, 4, 32, 32))
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.features[0][0].weight.grad[:, 3].abs().sum() > 0

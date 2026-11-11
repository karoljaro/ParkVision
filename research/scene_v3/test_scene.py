"""Guard real mask supervision, semantic roles and non-exclusive ROI evidence."""

from types import SimpleNamespace

import numpy as np
from pycocotools import mask as coco_mask
import pytest

from research.scene_v3.data import VEHICLE_NAMES, mask_array, source_group
from research.scene_v3.model import mask_supervision, load_checkpoint
from research.scene_v3.evidence import overlaps


def test_real_rle_and_polygon_mask_decode():
    mask = np.zeros((20, 30), np.uint8)
    mask[3:12, 5:18] = 1
    encoded = coco_mask.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")
    assert np.array_equal(mask_array({"segmentation": encoded}, 20, 30), mask)
    polygon = {"segmentation": [[5, 3, 18, 3, 18, 12, 5, 12]]}
    assert np.array_equal(mask_array(polygon, 20, 30), mask)


def test_bbox_is_never_converted_to_fake_mask_and_rle_size_is_checked():
    with pytest.raises(ValueError, match="bounding box"):
        mask_array({"bbox": [1, 2, 3, 4], "segmentation": []}, 10, 10)
    with pytest.raises(ValueError, match="dimensions"):
        mask_array({"segmentation": {"size": [20, 20], "counts": "abc"}}, 10, 10)


def test_parking_state_classes_are_not_vehicle_supervision():
    assert not VEHICLE_NAMES & {
        "Vacant",
        "Occupied",
        "free",
        "empty",
        "parking space",
        "parkinglane",
    }


def test_city_group_keeps_all_tiles_and_rotations_together():
    assert (
        source_group("dop10rgbi_tile_1", ["Herne"])[0]
        == source_group("dop10rgbi_tile_999", ["Herne"])[0]
    )
    assert source_group("IMG_0378_3-0017")[0] == source_group("IMG_0378_3-0087")[0]
    assert source_group("frame_17")[1] == "unknown"


def test_mask_branch_disabled_for_bbox_only_and_restored_on_failure():
    head = object()
    model = SimpleNamespace(roi_heads=SimpleNamespace(mask_predictor=head))
    with pytest.raises(RuntimeError):
        with mask_supervision(model, False):
            assert model.roi_heads.mask_predictor is None
            raise RuntimeError("simulated failure")
    assert model.roi_heads.mask_predictor is head


def projection(identity, left, right):
    return SimpleNamespace(
        parking_slot_id=identity,
        polygon=SimpleNamespace(
            vertices=[
                SimpleNamespace(x=left, y=0.0),
                SimpleNamespace(x=right, y=0.0),
                SimpleNamespace(x=right, y=1.0),
                SimpleNamespace(x=left, y=1.0),
            ]
        ),
    )


def test_one_vehicle_can_support_two_slots_without_forced_assignment():
    mask = np.ones((20, 40), np.float32)
    instances, slots = overlaps(
        [projection("A", 0, 0.45), projection("B", 0.55, 1)], [mask], [0.99], 40, 20
    )
    assert instances[0]["multi_slot_support"]
    assert {r["slot_id"] for r in instances[0]["slot_overlaps"]} == {"A", "B"}
    assert all(
        r["occupancy_override"] is None and r["visibility"] == "unassessed"
        for r in slots
    )


def test_missing_vehicle_evidence_does_not_prove_free():
    instances, slots = overlaps([projection("A", 0, 1)], [], [], 40, 20)
    assert not instances and slots[0]["no_instance_does_not_establish_free"]
    assert slots[0]["occupancy_override"] is None


def test_invalid_checkpoint_hash_rejected_before_deserialization(tmp_path):
    weights = tmp_path / "bad.pt"
    weights.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(weights, "0" * 64, "cpu")


def test_unlabelled_map_channel_and_masked_pixels_have_zero_gradient():
    import torch
    from research.scene_v3.maps import map_loss

    logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
    target = torch.ones_like(logits)
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    valid[:, :2] = False
    supervised = torch.tensor([[False, True]])
    map_loss(logits, target, valid, supervised).backward()
    assert logits.grad[:, 0].abs().sum() == 0
    assert logits.grad[:, :, :2].abs().sum() == 0
    assert logits.grad[:, 1, 2:].abs().sum() > 0


def test_empty_raster_polygon_is_not_silently_a_negative_target():
    with pytest.raises(ValueError, match="Empty"):
        mask_array({"segmentation": [[1.0, 1.0, 1.01, 1.0, 1.0, 1.01]]}, 10, 10)


def test_map_letterbox_preserves_target_and_ignores_padding():
    from research.scene_v3.maps import prepare_input

    rgb = np.full((100, 200, 3), 100, np.uint8)
    masks = np.ones((2, 100, 200), np.uint8)
    tensor, target, valid, (x, y, w, h) = prepare_input(rgb, masks)
    assert tensor.shape == (3, 512, 512)
    assert (x, y, w, h) == (0, 128, 512, 256)
    assert not valid[:128].any() and valid[128:384].all()
    assert target[:, 128:384].sum() == 2 * 512 * 256

"""Synthetic contract tests; no dataset test images or ImageNet downloads."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Optional training environment required")

import train  # noqa: E402
from torch import nn  # noqa: E402
from overfit import Tile, tensors  # noqa: E402
from prepare import convert, initialize, read_json, sha256, write_json  # noqa: E402


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    sources = []
    for i, split in enumerate(("train", "val", "test")):
        path = tmp_path / f"{split}.png"
        assert cv2.imwrite(str(path), np.full((64, 96, 3), 60 + i * 40, np.uint8))
        sources.append(path)
    initialize(sources, tmp_path / "inputs")
    manifest_path = tmp_path / "inputs/manifest.json"
    manifest = read_json(manifest_path)
    for sample in manifest["samples"]:
        split = sample["id"]
        sample.update(
            parking_id=f"site-{split}",
            camera_id=f"camera-{split}",
            group_id=f"day-{split}",
            source="synthetic-unit-test",
            license="synthetic-unit-test",
        )
        annotation: dict = {
            "flags": {"reviewed": True},
            "imageData": None,
            "imagePath": f"{split}.png",
            "imageHeight": 64,
            "imageWidth": 96,
        }
        annotation["shapes"] = [
            {
                "label": name,
                "shape_type": "polygon",
                "points": [[x, 2], [x + 10, 2], [x + 10, 30], [x, 30]],
            }
            for name, x in (("background", 2), ("paint", 20), ("curb", 40))
        ]
        write_json(tmp_path / "inputs" / sample["annotation"], annotation)
    write_json(manifest_path, manifest)
    convert(manifest_path, tmp_path / "export")
    for sample in manifest["samples"]:
        sample["split"] = sample["id"]
        for key in ("image", "annotation"):
            sample[key] = str((manifest_path.parent / sample[key]).resolve())
    split_path = tmp_path / "splits.json"
    write_json(split_path, manifest)
    root = tmp_path / "frozen"
    train.freeze_dataset(tmp_path / "export/dataset.json", split_path, root)
    return root


def test_frozen_split_copies_and_hashes_are_checked(frozen: Path) -> None:
    data = train.verify_freeze(frozen)
    assert {s["split"] for s in data["samples"]} == {"train", "val", "test"}
    for sample in data["samples"]:
        assert sample["mask_sha256"] == sha256(frozen / sample["mask"])
    with pytest.raises(ValueError, match="NEW directory"):
        train.freeze_dataset(frozen / "dataset.json", frozen / "dataset.json", frozen)
    sample = data["samples"][0]
    path = frozen / sample["annotation"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen input changed"):
        train.verify_freeze(frozen)


def test_training_loaders_cannot_decode_test_or_mix_splits(
    frozen: Path, monkeypatch
) -> None:
    original = train.checked_sample
    seen = []

    def checked(sample, root):
        assert sample["split"] != "test"
        seen.append(sample["split"])
        return original(sample, root)

    monkeypatch.setattr(train, "checked_sample", checked)
    train.load_split(frozen, "train", 64)
    train.load_split(frozen, "val", 64)
    assert seen == ["train", "val"]
    with pytest.raises(ValueError, match="sealed"):
        train.load_split(frozen, "test", 64)


def test_freeze_refuses_views_of_one_parking_in_different_splits(
    frozen: Path, tmp_path: Path
) -> None:
    proposal = read_json(frozen / "dataset.json")
    train_sample = next(s for s in proposal["samples"] if s["split"] == "train")
    test_sample = next(s for s in proposal["samples"] if s["split"] == "test")
    test_sample["parking_id"] = train_sample["parking_id"]
    for sample in proposal["samples"]:
        for key in ("image", "annotation"):
            sample[key] = str(frozen / sample[key])
    split_path = tmp_path / "leaking-splits.json"
    write_json(split_path, proposal)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Split leakage for parking_id"):
        train.freeze_dataset(frozen / "dataset.json", split_path, output)
    assert not output.exists()


def test_polygon_agreement_required_even_with_changed_mask_hash(frozen: Path) -> None:
    data = read_json(frozen / "dataset.json")
    sample = data["samples"][0]
    path = frozen / sample["mask"]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    mask[0, 0] = 1
    assert cv2.imwrite(str(path), mask)
    sample["mask_sha256"] = sha256(path)
    write_json(frozen / "dataset.json", data)
    with pytest.raises(ValueError, match="does not match reviewed polygons"):
        train.load_split(frozen, sample["split"], 64)


def test_metrics_exact_values_and_ignore_invariance() -> None:
    target = torch.tensor([0, 0, 1, 1, 2, 2, 255, 255])
    pred = torch.tensor([0, 1, 1, 0, 2, 1, 0, 2])
    matrix = train.confusion_matrix(pred, target)
    assert matrix.tolist() == [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    report = train.metrics_from_confusion(matrix)
    assert report["scored_pixels"] == 6
    assert report["classes"]["paint"]["iou"] == pytest.approx(1 / 4)
    assert report["classes"]["paint"]["dice"] == pytest.approx(2 / 5)
    assert report["classes"]["paint"]["precision"] == pytest.approx(1 / 3)
    assert report["classes"]["paint"]["recall"] == pytest.approx(1 / 2)
    pred[-2:] = torch.tensor([2, 0])
    assert torch.equal(matrix, train.confusion_matrix(pred, target))


def test_undefined_metrics_are_not_artificially_perfect() -> None:
    report = train.metrics_from_confusion(torch.diag(torch.tensor([5, 0, 0])))
    assert report["classes"]["paint"]["precision"] is None
    assert report["classes"]["paint"]["recall"] is None
    assert report["classes"]["paint"]["iou"] is None
    assert report["foreground_macro_iou"] is None


def test_augmentation_preserves_class_counts_and_paired_geometry() -> None:
    mask = np.resize(np.array([0, 1, 2, 255], np.uint8), (64, 64))
    image = np.repeat((mask % 3)[:, :, None] * 80, 3, axis=2)
    tile = Tile("synthetic", 0, 0, image, mask)
    original_mask = mask.copy()
    for seed in range(8):
        transformed = train.augment(tile, random.Random(seed))
        np.testing.assert_array_equal(
            np.unique(transformed.mask, return_counts=True)[1],
            np.unique(mask, return_counts=True)[1],
        )
        # Every class remains associated with exactly one image intensity despite
        # rotation, reflection and image-only brightness/contrast.
        for value in (0, 1, 2, 255):
            assert len(np.unique(transformed.image[transformed.mask == value, 0])) == 1
    np.testing.assert_array_equal(mask, original_mask)
    images, targets = tensors([transformed])
    logits = torch.randn(1, 3, 64, 64, requires_grad=True)
    loss = train.training_loss(logits, targets, torch.ones(3))
    loss.backward()
    assert torch.all(logits.grad.permute(0, 2, 3, 1)[targets == 255] == 0)
    assert images.shape == (1, 3, 64, 64)


def test_early_stopping_significant_improvements_and_patience() -> None:
    stopping = train.EarlyStopping(patience=2, min_delta=0.01)
    assert not stopping.update(0.2)
    assert not stopping.update(0.205)
    assert not stopping.update(0.23)
    assert not stopping.update(0.22)
    assert stopping.update(0.229)


@pytest.mark.parametrize("foreground", [True, False])
def test_loss_matches_original_ce_dice_value_and_gradient(foreground: bool) -> None:
    from overfit import training_loss as reference

    targets = torch.tensor([[[0, 1, 255], [2, 255, 0]]])
    if not foreground:
        targets[targets != 255] = 0
    weights = torch.tensor([0.3, 1.2, 1.5])
    logits = torch.randn(1, 3, 2, 3, requires_grad=True)
    other = logits.detach().clone().requires_grad_()
    actual = train.training_loss(logits, targets, weights)
    expected = reference(other, targets, weights)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, other.grad)
    assert torch.all(logits.grad.permute(0, 2, 3, 1)[targets == 255] == 0)
    with pytest.raises(ValueError, match="reviewed pixels"):
        train.training_loss(logits, torch.full_like(targets, 255), weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not accessible")
def test_loss_supports_strict_deterministic_cuda() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        logits = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
        target = torch.randint(0, 3, (2, 32, 32), device="cuda")
        target[:, :8] = 255
        loss = train.training_loss(logits, target, torch.ones(3, device="cuda"))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.all(logits.grad.permute(0, 2, 3, 1)[target == 255] == 0)
    finally:
        torch.use_deterministic_algorithms(previous)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 3, 1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def test_short_training_checkpoint_reload_and_only_val_previews(
    frozen: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(train, "build_model", lambda pretrained=None: TinyModel())
    pretrained = tmp_path / "synthetic-pretrained.json"
    write_json(pretrained, {"test_only": True})
    args = argparse.Namespace(
        frozen=frozen,
        pretrained=pretrained,
        output=tmp_path / "run",
        device="cpu",
        epochs=2,
        patience=2,
        min_delta=0.001,
        tile_size=64,
        batch_size=2,
        seed=42,
    )
    report = train.run(args)
    assert report["checkpoint_reload_predictions_identical"]
    assert not report["test_evaluated"]
    assert report["validation"]["scored_pixels"] == 3 * 11 * 29
    assert set(report["validation"]["per_image"]) == {"val"}
    checkpoint = torch.load(
        args.output / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["config"]["train_ids"] == ["train"]
    assert (args.output / "validation/val.roi-prediction.png").is_file()
    assert not list((args.output / "validation").glob("test*"))
    with pytest.raises(ValueError, match="NEW directory"):
        train.run(args)

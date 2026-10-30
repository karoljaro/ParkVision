"""Training data integrity and ignore semantics; no real images become test labels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Optional overfit environment required")

from overfit import fit_passed, load_tiles, native_tiles, training_loss  # noqa: E402
from prepare import convert, initialize, read_json, sha256, write_json  # noqa: E402


@pytest.fixture
def pilot_export(tmp_path: Path) -> Path:
    image = np.full((12, 10, 3), 90, np.uint8)
    source = tmp_path / "frame.png"
    assert cv2.imwrite(str(source), image)
    pilot = tmp_path / "pilot"
    initialize([source], pilot)
    shapes = []
    for label, x in [("background", 0), ("paint", 3), ("curb", 6)]:
        shapes.append(
            {
                "label": label,
                "shape_type": "polygon",
                "points": [[x, 1], [x + 1, 1], [x + 1, 8], [x, 8]],
            }
        )
    write_json(
        pilot / "images/frame.json",
        {
            "flags": {"reviewed": True},
            "imageData": None,
            "imagePath": "frame.png",
            "imageHeight": 12,
            "imageWidth": 10,
            "shapes": shapes,
        },
    )
    convert(pilot / "manifest.json", tmp_path / "export")
    return tmp_path / "export/dataset.json"


def test_native_grid_preserves_pixels_and_padding_is_ignore() -> None:
    image = np.arange(35 * 67 * 3, dtype=np.uint8).reshape(35, 67, 3)
    mask = np.full((35, 67), 255, np.uint8)
    mask[3:35, 29:67] = 0
    mask[33, 64] = 1
    mask[0, 0] = 2
    tiles = list(native_tiles("sample", image, mask, 32))
    restored = np.full_like(mask, 255)
    for tile in tiles:
        h = min(32, image.shape[0] - tile.y)
        w = min(32, image.shape[1] - tile.x)
        np.testing.assert_array_equal(
            tile.image[:h, :w], image[tile.y : tile.y + h, tile.x : tile.x + w]
        )
        assert np.all(tile.mask[h:] == 255)
        assert np.all(tile.mask[:, w:] == 255)
        restored[tile.y : tile.y + h, tile.x : tile.x + w] = tile.mask[:h, :w]
    np.testing.assert_array_equal(restored, mask)
    assert sum(int(np.count_nonzero(t.mask != 255)) for t in tiles) == int(
        np.count_nonzero(mask != 255)
    )


@pytest.mark.parametrize("size", [0, -32, 31, 33])
def test_invalid_tile_size_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of 32"):
        list(
            native_tiles(
                "sample",
                np.zeros((1, 1, 3), np.uint8),
                np.zeros((1, 1), np.uint8),
                size,
            )
        )


def test_ignore_logits_do_not_change_loss_or_receive_gradients() -> None:
    target = torch.tensor([[[0, 1, 255], [2, 255, 0]]])
    logits = torch.randn(1, 3, 2, 3, requires_grad=True)
    weights = torch.tensor([0.3, 1.2, 1.5])
    baseline = training_loss(logits, target, weights)
    changed = logits.detach().clone()
    changed[:, :, target[0] == 255] = torch.tensor([1000.0, -1000.0, 2000.0])[:, None]
    torch.testing.assert_close(training_loss(changed, target, weights), baseline)
    baseline.backward()
    assert torch.equal(logits.grad[:, :, target[0] == 255], torch.zeros(1, 3, 2))
    assert logits.grad[:, :, target[0] != 255].abs().sum() > 0


def test_all_ignore_batch_rejected_instead_of_nan() -> None:
    with pytest.raises(ValueError, match="reviewed pixels"):
        training_loss(
            torch.zeros(1, 3, 2, 2), torch.full((1, 2, 2), 255), torch.ones(3)
        )


def test_negative_only_batch_has_no_artificial_foreground_dice_penalty() -> None:
    logits = torch.randn(1, 3, 2, 2)
    target = torch.tensor([[[0, 255], [0, 255]]])
    weights = torch.ones(3)
    expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    torch.testing.assert_close(training_loss(logits, target, weights), expected)


def test_overfit_cannot_pass_by_overpredicting_a_foreground_class() -> None:
    initial = {"training_loss": 3.0}
    precision = [0.99, 0.98, 0.50]
    current = {
        "training_loss": 0.1,
        "training_pixel_recall": [0.96, 0.98, 0.99],
        "training_pixel_precision": precision,
    }
    assert not fit_passed(initial, current)
    precision[2] = 0.95
    assert fit_passed(initial, current)


def test_loader_accepts_reviewed_export_without_mutating_sources(
    pilot_export: Path,
) -> None:
    dataset = read_json(pilot_export)
    annotation = Path(dataset["samples"][0]["annotation"])
    before = sha256(annotation)
    tiles, _ = load_tiles(pilot_export, 32)
    assert len(tiles) == 1
    assert sha256(annotation) == before
    assert np.count_nonzero(tiles[0].mask != 255) == 48


@pytest.mark.parametrize("kind", ["annotation", "mask", "image"])
def test_loader_rejects_edited_source_after_export(
    pilot_export: Path, kind: str
) -> None:
    sample = read_json(pilot_export)["samples"][0]
    path = Path(sample[kind]) if kind != "mask" else pilot_export.parent / sample[kind]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after export|sha256 mismatch"):
        load_tiles(pilot_export, 32)


def test_reviewed_flag_is_required_even_with_updated_hash(pilot_export: Path) -> None:
    dataset = read_json(pilot_export)
    sample = dataset["samples"][0]
    annotation = Path(sample["annotation"])
    data = read_json(annotation)
    data["flags"]["reviewed"] = False
    write_json(annotation, data)
    sample["annotation_sha256"] = sha256(annotation)
    write_json(pilot_export, dataset)
    with pytest.raises(ValueError, match="reviewed flag"):
        load_tiles(pilot_export, 32)


def test_changed_mask_cannot_be_reblessed_by_just_updating_hash(
    pilot_export: Path,
) -> None:
    dataset = read_json(pilot_export)
    sample = dataset["samples"][0]
    path = pilot_export.parent / sample["mask"]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    mask[0, 0] = 1
    assert cv2.imwrite(str(path), mask)
    sample["mask_sha256"] = sha256(path)
    write_json(pilot_export, dataset)
    with pytest.raises(ValueError, match="does not match the reviewed annotation"):
        load_tiles(pilot_export, 32)


def test_nonpilot_split_is_not_silently_used_for_training(pilot_export: Path) -> None:
    dataset = read_json(pilot_export)
    sample = dataset["samples"][0]
    sample.update(
        split="test",
        parking_id="site",
        camera_id="site:camera",
        group_id="session",
        license="test-only",
    )
    write_json(pilot_export, dataset)
    with pytest.raises(ValueError, match="only split=pilot"):
        load_tiles(pilot_export, 32)


def test_class_ids_cannot_be_silently_remapped(pilot_export: Path) -> None:
    dataset = read_json(pilot_export)
    dataset["classes"]["paint"] = 2
    write_json(pilot_export, dataset)
    with pytest.raises(ValueError, match="class mapping"):
        load_tiles(pilot_export, 32)

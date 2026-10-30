"""Small artificial rasters test data mechanics, never act as training labels."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from prepare import (
    check_leakage,
    convert,
    initialize,
    inspect_manifest,
    labelme_mask,
    load_manifest,
    overlay,
    read_image,
    read_json,
    sha256,
    validate_mask,
    write_json,
)


def polygon(label: str, points: list[list[float]]) -> dict:
    return {"label": label, "shape_type": "polygon", "points": points, "flags": {}}


def rectangle(label: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return polygon(label, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def annotation() -> dict:
    return {
        "version": "5.9.1",
        "flags": {"reviewed": True},
        "imagePath": "frame.png",
        "imageData": None,
        "imageWidth": 10,
        "imageHeight": 8,
        "shapes": [
            rectangle("background", 0, 0, 5, 7),
            rectangle("paint", 1, 1, 2, 5),
            rectangle("curb", 4, 2, 5, 4),
            rectangle("ignore", 1, 3, 2, 4),
        ],
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), image)
    write_json(tmp_path / "frame.json", annotation())
    manifest = {
        "version": 1,
        "samples": [
            {
                "id": "frame",
                "image": "frame.png",
                "annotation": "frame.json",
                "parking_id": "site-a",
                "camera_id": "site-a:camera-1",
                "group_id": "site-a:capture-a",
                "split": "train",
                "surface": "wet",
                "conditions": ["reflection"],
                "source": "test-only-artificial-raster",
                "license": "test-fixture",
                "image_sha256": sha256(image_path),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return path


def test_exact_pixels_unknown_exterior_and_order_independence() -> None:
    image = np.zeros((8, 10, 3), np.uint8)
    data = annotation()
    mask = labelme_mask(data, image)
    expected = np.full((8, 10), 255, np.uint8)
    expected[:, :6] = 0
    expected[1:6, 1:3] = 1
    expected[2:5, 4:6] = 2
    expected[3:5, 1:3] = 255
    np.testing.assert_array_equal(mask, expected)
    data["shapes"].reverse()
    np.testing.assert_array_equal(labelme_mask(data, image), expected)
    assert not image.any()


def test_overlay_keeps_source_and_unlabelled_pixels() -> None:
    image = np.full((8, 10, 3), 100, np.uint8)
    original = image.copy()
    image.flags.writeable = False
    mask = labelme_mask(annotation(), image)
    result = overlay(image, mask)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(result[mask == 255], original[mask == 255])
    np.testing.assert_array_equal(result[1, 1], [178, 50, 178])


@pytest.mark.parametrize(
    "field,value",
    [
        ("imageWidth", 11),
        ("imageHeight", 7),
        ("imageData", "stale-data"),
        ("flags", {"reviewed": False}),
        ("flags", {}),
        ("flags", None),
        ("shapes", []),
        ("shapes", None),
    ],
)
def test_reject_incomplete_or_incompatible_annotation(
    field: str, value: object
) -> None:
    data = annotation()
    data[field] = value
    with pytest.raises(ValueError):
        labelme_mask(data, np.zeros((8, 10, 3), np.uint8))


@pytest.mark.parametrize(
    "update",
    [
        {"label": "slot"},
        {"shape_type": "line"},
        {"shape_type": "rectangle"},
        {"points": [[0, 0], [2, 2]]},
        {"points": [[0, 0], [1, 1], [2, 2]]},
        {"points": [[0, 0], [10, 1], [2, 2]]},
        {"points": [[-1, 0], [2, 1], [2, 2]]},
        {"points": [[0, 0], [float("nan"), 1], [2, 2]]},
        {"points": [[0, 0], [float("inf"), 1], [2, 2]]},
        {"mask": "unsupported"},
    ],
)
def test_reject_unsupported_shapes(update: dict) -> None:
    data = annotation()
    data["shapes"][0].update(update)
    with pytest.raises(ValueError):
        labelme_mask(data, np.zeros((8, 10, 3), np.uint8))


def test_overlapping_paint_curb_requires_ignore() -> None:
    data = annotation()
    data["shapes"] += [rectangle("curb", 1, 1, 2, 5)]
    with pytest.raises(ValueError, match="overlap"):
        labelme_mask(data, np.zeros((8, 10, 3), np.uint8))
    data["shapes"] += [rectangle("ignore", 1, 1, 2, 5)]
    assert labelme_mask(data, np.zeros((8, 10, 3), np.uint8))[1, 1] == 255


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 10), np.int32),
        np.zeros((10, 8), np.uint8),
        np.zeros((8, 10, 3), np.uint8),
        np.full((8, 10), 3, np.uint8),
        np.full((8, 10), 255, np.uint8),
    ],
)
def test_reject_invalid_masks(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_mask(mask, (8, 10))


def test_export_round_trip_paths_hashes_and_no_overwrite(dataset: Path) -> None:
    before = {p.name: p.read_bytes() for p in dataset.parent.iterdir()}
    output = dataset.parent / "export"
    report = convert(dataset, output)
    record = report["samples"][0]
    source = read_image(dataset.parent / "frame.png")
    mask = cv2.imread(str(output / record["mask"]), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    np.testing.assert_array_equal(mask, labelme_mask(annotation(), source))
    assert mask.dtype == np.uint8 and mask.shape == (8, 10)
    assert record["pixels"] == {"background": 32, "paint": 6, "curb": 6, "ignore": 36}
    assert record["image_sha256"] == sha256(dataset.parent / "frame.png")
    assert record["mask_sha256"] == sha256(output / record["mask"])
    assert report == read_json(output / "dataset.json")
    with pytest.raises(FileExistsError):
        convert(dataset, output)
    assert all(
        (dataset.parent / name).read_bytes() == data for name, data in before.items()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../bad"),
        ("split", "validation"),
        ("surface", "rain"),
        ("parking_id", "unknown"),
        ("license", "unknown"),
        ("conditions", "wet"),
        ("image_sha256", "bad"),
    ],
)
def test_manifest_contract(dataset: Path, field: str, value: object) -> None:
    data = read_json(dataset)
    data["samples"][0][field] = value
    write_json(dataset, data)
    with pytest.raises(ValueError):
        load_manifest(dataset)


@pytest.mark.parametrize("field", ["parking_id", "camera_id", "group_id", "pixels"])
@pytest.mark.parametrize(
    "splits", [("train", "val"), ("train", "test"), ("val", "test")]
)
def test_blocks_cross_split_leakage(
    dataset: Path, field: str, splits: tuple[str, str]
) -> None:
    samples = load_manifest(dataset)
    samples[0]["split"] = splits[0]
    other = copy.deepcopy(samples[0])
    other.update(
        id="other",
        image="other.png",
        split=splits[1],
        parking_id="site-b",
        camera_id="site-b:camera-1",
        group_id="other",
    )
    image = read_image(dataset.parent / "frame.png")
    if field != "pixels":
        image[0, 0] = 0
        other[field] = samples[0][field]
    assert cv2.imwrite(str(dataset.parent / "other.png"), image)
    other["image_sha256"] = sha256(dataset.parent / "other.png")
    with pytest.raises(ValueError, match="Split leakage"):
        check_leakage([*samples, other], dataset.parent)


def test_accepts_distinct_sites_and_rejects_changed_source(dataset: Path) -> None:
    samples = load_manifest(dataset)
    other = copy.deepcopy(samples[0])
    other.update(
        id="other",
        image="other.png",
        split="val",
        parking_id="site-b",
        camera_id="site-b:camera-1",
        group_id="site-b:day-1",
    )
    assert cv2.imwrite(
        str(dataset.parent / "other.png"), np.zeros((8, 10, 3), np.uint8)
    )
    other["image_sha256"] = sha256(dataset.parent / "other.png")
    check_leakage([*samples, other], dataset.parent)
    with (dataset.parent / "frame.png").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="image_sha256 mismatch"):
        check_leakage(samples, dataset.parent)


@pytest.mark.parametrize("change", ["missing", "wrong-image", "unreviewed"])
def test_failed_batch_does_not_create_outputs(dataset: Path, change: str) -> None:
    annotation_path = dataset.parent / "frame.json"
    if change == "missing":
        annotation_path.unlink()
    else:
        data = read_json(annotation_path)
        if change == "wrong-image":
            data["imagePath"] = "other.png"
        else:
            data["flags"]["reviewed"] = False
        write_json(annotation_path, data)
    output = dataset.parent / "export"
    with pytest.raises((OSError, ValueError)):
        convert(dataset, output)
    assert not output.exists()


def test_initialize_real_bytes_without_fabricating_labels(dataset: Path) -> None:
    source = dataset.parent / "frame.png"
    output = dataset.parent / "pilot"
    initialize([source], output)
    assert (output / "images/frame.png").read_bytes() == source.read_bytes()
    assert not (output / "images/frame.json").exists()
    records = inspect_manifest(output / "manifest.json")["samples"]
    assert records[0]["annotation_exists"] is False
    assert records[0]["split"] == "pilot"
    with pytest.raises(FileExistsError):
        initialize([source], output)


def test_cli_from_another_directory(dataset: Path) -> None:
    script = Path(__file__).with_name("prepare.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "convert",
            str(dataset),
            "--output",
            str(dataset.parent / "export"),
        ],
        cwd="/tmp",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Exported 1 masks" in result.stdout
    data = json.loads((dataset.parent / "export/dataset.json").read_text())
    assert data["samples"][0]["pixels"]["paint"] == 6

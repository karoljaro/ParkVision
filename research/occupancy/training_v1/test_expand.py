"""Expansion preserves immutable inputs and flags data hygiene problems."""

import json
import cv2
import numpy as np
import pytest

from .data import file_hash
from .expand import clone_base, inspect_images


def test_clone_is_byte_identical_and_rejects_changed_inherited_files(tmp_path):
    base, output = tmp_path / "base", tmp_path / "new"
    base.mkdir()
    (base / "image.jpg").write_bytes(b"source image")
    (base / "array.npy").write_bytes(b"source array")
    (base / "source.json").write_bytes(b"{}")
    manifest = {
        "frames": [{"image": "image.jpg", "sha256": file_hash(base / "image.jpg")}],
        "samples": [{"array": "array.npy", "sha256": file_hash(base / "array.npy")}],
        "sources": {
            "source": {
                "archive": "source.json",
                "sha256": file_hash(base / "source.json"),
            }
        },
    }
    before = json.dumps(manifest)
    clone_base(base, output, manifest)
    assert json.dumps(manifest) == before
    for name in ("image.jpg", "array.npy", "source.json"):
        assert (base / name).read_bytes() == (output / name).read_bytes()
    # Replace the destination inode so the immutable source is not modified.
    (output / "array.npy").unlink()
    (output / "array.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing expanded"):
        clone_base(base, output, manifest)


def test_visual_duplicate_across_roles_is_flagged(tmp_path):
    image = np.zeros((50, 60, 3), np.uint8)
    image[:, 20:40] = 200
    rows = []
    for split in ("train", "val"):
        name = split + ".jpg"
        cv2.imwrite(str(tmp_path / name), image)
        rows.append({"image": name, "source_path": name, "split": split})
    result = inspect_images(tmp_path, {"frames": rows})
    assert result["cross_split_near_candidates"] == [
        {"train": "train.jpg", "val": "val.jpg", "hamming_256": 0}
    ]
    assert result["exif_orientations"] == {"1": 2}

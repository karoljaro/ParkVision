"""Local export staging must not relabel free boxes or trust exporter splits."""

import json
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from .data import file_hash
from .roboflow import (
    audit,
    canonical_name,
    checked_member,
    NearIndex,
    source_family,
    task_for,
    verify_prepared,
)


def test_semantic_mapping_keeps_occupancy_out_of_incompatible_tasks():
    assert (
        task_for([{"name": "parking_space"}, {"name": "blocked_space"}])
        == "slot_geometry_only"
    )
    assert task_for([{"name": "ParkingLines"}]) == "paint_positive_polygons_only"
    assert (
        task_for([{"name": "car"}, {"name": "free"}])
        == "vehicle_and_free_region_boxes_only"
    )


def test_filename_variants_share_source_and_video_family():
    a = canonical_name({"file_name": "clip_mp4-0001_jpg.rf.aabb.jpg"})
    b = canonical_name({"file_name": "clip_mp4-0002_jpg.rf.ccdd.jpg"})
    assert source_family(a) == source_family(b)
    assert canonical_name({"file_name": "clip_mp4-0001_jpg.rf.ffee.jpg"}) == a


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "dir\\escape.jpg"])
def test_zip_path_traversal_is_rejected(name):
    with pytest.raises(ValueError):
        checked_member(name)


def test_near_index_reports_candidates_without_declaring_identity():
    index = NearIndex()
    assert index.query_add(0, "a") == []
    assert index.query_add(3, "b") == [("a", 2)]
    assert index.query_add((1 << 256) - 1, "c") == []


def test_complete_local_zip_audit_is_lossless_and_detects_split_duplicates(tmp_path):
    inputs, output = tmp_path / "inputs", tmp_path / "prepared"
    inputs.mkdir()
    archive = inputs / "export.zip"
    pixels = np.full((32, 40, 3), 120, np.uint8)
    jpeg = cv2.imencode(".jpg", pixels)[1].tobytes()
    with ZipFile(archive, "w") as package:
        for split in ("train", "valid", "test"):
            filename = f"clip_mp4-0001_jpg.rf.{split}.jpg"
            data = {
                "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "free"}],
                "images": [{"id": 0, "file_name": filename, "width": 40, "height": 32}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 0,
                        "category_id": 2,
                        "bbox": [1, 1, 20, 20],
                        "segmentation": [],
                    }
                ],
            }
            package.writestr(f"{split}/_annotations.coco.json", json.dumps(data))
            package.writestr(f"{split}/{filename}", jpeg)
    before = file_hash(archive)
    result = audit(inputs, output)
    assert result["images"] == 3
    assert result["unique_raw_images"] == 1
    assert result["cross_split_exact_groups"] == 1
    assert file_hash(archive) == before
    rows = [
        json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()
    ]
    assert all(
        row["occupancy_label"] is None and not row["training_eligible"] for row in rows
    )
    assert [row["staging_role"] for row in rows if row["split"] == "test"] == [
        "heldout"
    ]
    assert (output / rows[0]["file"]).read_bytes() == jpeg
    verify_prepared(output)
    (output / rows[0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Staged image changed"):
        verify_prepared(output)

"""Real preparation semantics on synthetic images, never benchmark tuning."""

import copy
import json

import cv2
import numpy as np
import pytest

from research.occupancy.training_v1.data import file_hash
from research.occupancy.visibility_v2.dataset import (
    apply_labels,
    build,
    historical_visibility,
    quarantine_duplicate_groups,
    verify,
)


def test_duplicate_families_are_transitive_and_include_different_rois_of_one_frame():
    records = [
        {"id": "a", "frame_pixel_sha256": "frame-a", "split": "train"},
        {"id": "b", "frame_pixel_sha256": "frame-b", "split": "train"},
        {"id": "c", "frame_pixel_sha256": "frame-c", "split": "val"},
        {"id": "d", "frame_pixel_sha256": "frame-c", "split": "val"},
    ]
    quarantine_duplicate_groups(records, [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}])
    assert {r["split"] for r in records} == {"quarantine"}


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "source.png"
    rng = np.random.default_rng(23)
    cv2.imwrite(str(image), rng.integers(0, 256, (100, 140, 3), dtype=np.uint8))
    return {
        "image_path": str(image),
        "image_sha256": file_hash(image),
        "source_path": "PKLot/UFPR04/Cloudy/2012-12-01/frame.jpg",
        "roi_normalized": [[0.1, 0.2], [0.4, 0.2], [0.4, 0.6], [0.1, 0.6]],
        "region_role": "physical_parking_slot",
        "site_id": "ufpr",
        "camera_id": "UFPR04",
        "day_id": "2012-12-01",
        "slot_projection_id": "UFPR04:slot1",
        "physical_slot_id": None,
        "references": [],
        "visibility": "unassessed",
        "license": "synthetic test",
    }


def prepare(tmp_path, records):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site_splits": {"ufpr": "train", "cnr": "val"},
                "records": records,
            }
        )
    )
    output = tmp_path / "prepared"
    return build(path, output), output / "manifest.json"


def test_partial_requires_a_known_state_and_unassessed_is_not_clear():
    assert (
        historical_visibility({"visibility": "partial", "state": "free"})
        == "partial-but-readable"
    )
    assert (
        historical_visibility({"visibility": "partial", "state": "unknown"})
        == "unassessed"
    )
    assert (
        historical_visibility({"visibility": "unreadable", "state": "unknown"})
        == "unreadable"
    )


def test_manifest_provenance_ui_integrity_and_exclusive_outputs(tmp_path, source):
    result, manifest = prepare(tmp_path, [source])
    record = result["records"][0]
    assert record["visibility"] == "unassessed"
    assert record["split"] == "train" and not record["training_eligible"]
    assert record["provenance"] == [
        {k: v for k, v in source.items() if k != "visibility"}
    ]
    assert not verify(manifest)["training_ready"]
    review = (manifest.parent / "review.html").read_text()
    assert "__MANIFEST_DATA__" not in review and file_hash(manifest) in review
    assert "partial-but-readable" in review and "manifest_sha256" in review
    with pytest.raises(FileExistsError):
        build(tmp_path / "index.json", manifest.parent)
    (manifest.parent / record["crop"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="review image"):
        verify(manifest)


def test_cross_site_exact_duplicate_and_conflicting_labels_are_quarantined(
    tmp_path, source
):
    other = dict(source, site_id="cnr", visibility="unreadable")
    source["visibility"] = "clear"
    result, manifest = prepare(tmp_path, [source, other])
    assert result["dedup"]["exact_duplicates_merged"] == 1
    assert len(result["records"]) == 1
    row = result["records"][0]
    assert row["split"] == "quarantine" and row["label_conflict"]
    assert row["visibility"] == "unassessed" and len(row["provenance"]) == 2
    verify(manifest)


def test_unknown_site_and_tampered_source_are_not_accepted_as_train(tmp_path, source):
    source["site_id"] = None
    result, manifest = prepare(tmp_path, [source])
    assert result["records"][0]["split"] == "quarantine"
    array = manifest.parent / result["records"][0]["array"]
    array.write_bytes(b"broken")
    with pytest.raises((ValueError, EOFError)):
        verify(manifest)


@pytest.mark.parametrize("case", ["hash", "PUCPR", "T-01", "geometry"])
def test_protected_or_changed_inputs_fail(tmp_path, source, case):
    if case == "hash":
        source["image_sha256"] = "0" * 64
    elif case == "PUCPR":
        source["source_path"] = "PKLot/PUCPR/test.jpg"
    elif case == "T-01":
        source["region_role"] = "presence_only_nonparking"
    else:
        source["roi_normalized"][0][0] = 1.2
    with pytest.raises(ValueError):
        prepare(tmp_path, [source])


def test_operator_labels_bind_manifest_and_preserve_parent(tmp_path, source):
    result, manifest = prepare(tmp_path, [source])
    original = manifest.read_bytes()
    label_path = tmp_path / "labels.json"
    labels = {
        "operator": "test-operator",
        "manifest_sha256": file_hash(manifest),
        "records": {
            result["records"][0]["id"]: {
                "visibility": "unreadable",
                "note": "synthetic",
            }
        },
    }
    label_path.write_text(json.dumps(labels))
    revised = manifest.with_name("manifest-r2.json")
    apply_labels(manifest, label_path, revised)
    assert manifest.read_bytes() == original
    row = json.loads(revised.read_text())["records"][0]
    assert (
        row["visibility"] == "unreadable"
        and row["label_history"][0]["previous"] == "unassessed"
    )
    assert not verify(revised)["training_ready"]
    bad = copy.deepcopy(labels)
    bad["manifest_sha256"] = "changed"
    label_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="different manifest"):
        apply_labels(manifest, label_path, manifest.with_name("bad.json"))

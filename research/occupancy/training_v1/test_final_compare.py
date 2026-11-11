"""Synthetic checks only; never use held-out model scores to debug metrics."""

from copy import deepcopy
from hashlib import sha256
import json

import cv2
import numpy as np
import pytest

from research.occupancy import rules
from .comparison_metrics import (
    paired_changes,
    probability_metrics,
    reason_summary,
    summarize,
    visibility_gate,
)
from .data import file_hash
from .final_compare import acpds_frames, assignments, decode_and_check


def test_unknown_is_neither_free_nor_correct_for_known_binary_target():
    rows = [
        {"source_state": "free", "p": "free"},
        {"source_state": "free", "p": "unknown"},
        {"source_state": "occupied", "p": "free"},
        {"source_state": "occupied", "p": "occupied"},
        {"source_state": "unknown", "p": "occupied"},
        {"source_state": "unknown", "p": "unknown"},
    ]
    result = summarize(rows, "p")
    assert result["false_free"] == 1
    assert result["false_occupied"] == 0
    assert result["free_recall"] == result["occupied_recall"] == 0.5
    assert result["agreement_on_all_known"] == 0.5
    assert result["unknown"] == 2
    assert result["coverage"] == 4 / 6
    assert result["unsupported_decisions_on_unknown_truth"] == 1
    assert result["correct_unknown"] == 1


def test_no_positive_predictions_has_undefined_precision():
    result = summarize([{"source_state": "free", "p": "unknown"}], "p")
    assert result["free_precision_known"] is None
    assert result["occupied_precision_known"] is None
    assert result["free_recall"] == 0
    assert result["occupied_recall"] is None


def test_error_to_unknown_is_not_reported_as_correct():
    rows = [
        {"source_state": "occupied", "old": "free", "new": "unknown"},
        {"source_state": "free", "old": "unknown", "new": "free"},
    ]
    assert paired_changes(rows, "old", "new") == {
        "abstain->correct": 1,
        "wrong->abstain": 1,
    }


@pytest.mark.parametrize(
    "visibility,expected",
    [("clear", "free"), ("partial", "unknown"), ("unreadable", "unknown")],
)
def test_shared_visibility_gate_is_conservative(visibility, expected):
    assert visibility_gate("free", visibility) == expected


def test_conflict_counts_distinguish_correct_free_blocked_and_error_prevented():
    rows = [
        {"source_state": "free", "reason": "model_conflict"},
        {"source_state": "occupied", "reason": "model_conflict"},
        {"source_state": "free", "reason": "low_confidence"},
    ]
    result = reason_summary(rows, "reason")
    assert result["free_evidence_blocked_by_conflict"] == 1
    assert result["false_free_prevented_by_conflict"] == 1


def test_probability_metrics_are_finite_and_skip_unknown_truth():
    result = probability_metrics(
        [
            {"source_state": "free", "p": 0.0},
            {"source_state": "occupied", "p": 1.0},
            {"source_state": "unknown", "p": 0.5},
        ],
        "p",
    )
    assert result["n"] == 2 and result["accuracy_at_0_5"] == 1
    assert result["brier"] == 0
    assert np.isfinite(result["cross_entropy"])


def test_c_assignment_is_sensitive_and_d_rejects_weak_overlap():
    rois = [[[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]]]
    detection = {"bbox": [0, 0, 100, 100], "score": 0.9, "category": "car"}
    c, d = assignments(rules, rois, 101, 101, [detection])
    assert c["0"] == [0] and d["0"] == []
    assert rules.classify(0.05, c["0"]) == ("unknown", "model_conflict")
    assert rules.classify(0.05, d["0"]) == ("free", None)


def test_c_and_d_reject_equal_overlap_ties():
    rois = [
        [[0, 0], [0.25, 0], [0.25, 1], [0, 1]],
        [[0.75, 0], [1, 0], [1, 1], [0.75, 1]],
    ]
    detection = {"bbox": [0, 0, 100, 100], "score": 0.9, "category": "car"}
    c, d = assignments(rules, rois, 101, 101, [detection])
    assert all(not values for values in c.values())
    assert all(not values for values in d.values())


def test_test_image_cannot_overlap_training_even_after_reencoding(tmp_path):
    image = np.full((40, 50, 3), 120, np.uint8)
    raw = cv2.imencode(".png", image)[1].tobytes()
    (tmp_path / "frame.png").write_bytes(raw)
    frame = {
        "root": str(tmp_path),
        "image": "frame.png",
        "sha256": sha256(raw).hexdigest(),
        "rois": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
        "states": ["free"],
    }
    fingerprint = sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
    with pytest.raises(ValueError, match="Re-encoded"):
        decode_and_check(frame, set(), {fingerprint}, set(), set())
    with pytest.raises(ValueError, match="leakage"):
        decode_and_check(frame, {frame["sha256"]}, set(), set(), set())


def test_official_test_loader_rejects_train_image_and_changed_label(tmp_path):
    quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
    annotations = {
        "train": {"file_names": ["GOPR0001.JPG"]},
        "valid": {"file_names": ["GOPR0002.JPG"]},
        "test": {
            "file_names": ["GOPR0003.JPG"],
            "rois_list": [[quad]],
            "occupancy_list": [[False]],
        },
    }
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps(annotations))
    manifest = {
        "split": "official test",
        "annotations_sha256": file_hash(source),
        "frames": [{"image": "images/GOPR0003.JPG", "rois": [quad], "labels": [False]}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    protocol = {
        "files": {
            "acpds_manifest": {"path": str(path)},
            "acpds_annotations": {"path": str(source), "sha256": file_hash(source)},
        },
        "cohorts": {"ACPDS-test": {"root": str(tmp_path)}},
    }
    assert acpds_frames(protocol)[0]["states"] == ["free"]
    changed = deepcopy(manifest)
    changed["frames"][0]["image"] = "images/GOPR0001.JPG"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Unexpected"):
        acpds_frames(protocol)
    changed = deepcopy(manifest)
    changed["frames"][0]["labels"] = [True]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs"):
        acpds_frames(protocol)

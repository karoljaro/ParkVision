import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import observe
from models import load_assets
from rules import assign_vehicles, classify, intersection_area
from parkvision.configuration import load_configuration
from parkvision.domain.geometry.point import Point
from parkvision.domain.geometry.polygon import Polygon
from parkvision.domain.slot_projection import SlotProjection
from parkvision.domain.types import CameraId, ParkingSlotId

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "configuration.json"
    raw = (REPO / "apps/engine/examples/parking-setup-v1.json").read_bytes()
    path.write_bytes(raw)
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((720, 1280, 3), np.uint8))
    captured = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    return argparse.Namespace(
        configuration=path,
        camera_id=CameraId("camera-east"),
        session_id="test-session",
        sequence=4,
        image=image,
        captured_at=captured,
        evaluated_at=captured + timedelta(seconds=1),
        max_age_seconds=30,
        view_confirmed=True,
        unreadable_slot_id=[],
        assets=tmp_path / "assets.json",
        device="cpu",
    )


class FakeModels:
    provenance = {"fixture": True}
    scores = (0.1, 0.9)
    detections = ()

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, projections):
        assert frame.camera_id == "camera-east"
        assert frame.sequence == 4
        assert frame.image.shape == (720, 1280, 3)
        assert len(projections) == 2
        return self.scores, self.detections


def forbidden_models(*args, **kwargs):
    raise AssertionError("Invalid frame must never load models")


def test_success_reuses_ids_and_leaves_inputs_immutable(inputs):
    before_config = inputs.configuration.read_bytes()
    before_image = inputs.image.read_bytes()
    result = observe.observe(inputs, FakeModels)
    assert result["status"] == "observed"
    assert [(row["slot_id"], row["state"]) for row in result["observations"]] == [
        ("slot-01", "free"),
        ("slot-02", "occupied"),
    ]
    assert result["camera_id"] == "camera-east"
    assert result["source_session"] == "test-session"
    assert result["configuration_revision"] == 1
    assert result["provenance"]["models"] == {"fixture": True}
    assert result["captured_at"] == inputs.captured_at.isoformat()
    assert result["image"]["sha256"]
    assert inputs.configuration.read_bytes() == before_config
    assert inputs.image.read_bytes() == before_image


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "missing_frame"),
        ("empty", "invalid_frame"),
        ("broken", "invalid_frame"),
        ("dimensions", "dimension_mismatch"),
        ("view", "view_unverified"),
        ("time", "missing_capture_time"),
        ("stale", "stale_frame"),
        ("future", "future_frame"),
    ],
)
def test_bad_input_emits_unknown_without_inference(inputs, case, reason):
    if case == "missing":
        inputs.image.unlink()
    elif case == "empty":
        inputs.image.write_bytes(b"")
    elif case == "broken":
        inputs.image.write_bytes(b"not an image")
    elif case == "dimensions":
        assert cv2.imwrite(str(inputs.image), np.zeros((20, 20, 3), np.uint8))
    elif case == "view":
        inputs.view_confirmed = False
    elif case == "time":
        inputs.captured_at = None
    elif case == "stale":
        inputs.evaluated_at += timedelta(seconds=60)
    elif case == "future":
        inputs.evaluated_at = inputs.captured_at - timedelta(seconds=1)
    result = observe.observe(inputs, forbidden_models)
    assert result["status"] == reason
    assert all(
        row["state"] == "unknown" and row["unknown_reason"] == reason
        for row in result["observations"]
    )
    assert result["detections"] == []


def test_unknown_camera_and_unreadable_ids_are_input_errors(inputs):
    inputs.camera_id = CameraId("unconfigured")
    with pytest.raises(ValueError, match="Unknown camera"):
        observe.observe(inputs, forbidden_models)
    inputs.camera_id = CameraId("camera-east")
    inputs.unreadable_slot_id = ["other-camera-slot"]
    with pytest.raises(ValueError, match="Unreadable"):
        observe.observe(inputs, forbidden_models)


def test_many_camera_projection_uses_same_physical_slot(inputs):
    config = load_configuration(inputs.configuration)
    a = config.projections_for_camera(
        CameraId("camera-east"), image_width=1280, image_height=720
    )
    b = config.projections_for_camera(
        CameraId("camera-west"), image_width=1000, image_height=750
    )
    assert a[0].parking_slot_id == b[0].parking_slot_id == "slot-01"
    assert a[0].polygon != b[0].polygon


@pytest.mark.parametrize("scores", [(0.1,), (0.1, float("nan")), (0.1, 1.01)])
def test_late_model_error_invalidates_whole_frame(inputs, scores):
    class BrokenModels(FakeModels):
        pass

    BrokenModels.scores = scores
    result = observe.observe(inputs, BrokenModels)
    assert result["status"] == "inference_error"
    assert all(
        row["state"] == "unknown" and row["p_occupied"] is None
        for row in result["observations"]
    )
    assert result["detections"] == []


def test_missing_assets_is_reported_as_unknown(inputs):
    result = observe.observe(inputs)
    assert result["status"] == "inference_error"
    assert all(row["state"] == "unknown" for row in result["observations"])


def test_operator_unreadable_does_not_become_free(inputs):
    inputs.unreadable_slot_id = ["slot-01"]
    result = observe.observe(inputs, FakeModels)
    first, second = result["observations"]
    assert first["state"] == "unknown" and first["unknown_reason"] == "occluded"
    assert second["state"] == "occupied"


@pytest.mark.parametrize(
    "p,ids,expected",
    [
        (0.1, [], ("free", None)),
        (0.9, [], ("occupied", None)),
        (0.10001, [], ("unknown", "low_confidence")),
        (0.89999, [], ("unknown", "low_confidence")),
        (0.1, [0], ("unknown", "model_conflict")),
        (0.9, [0], ("occupied", None)),
        (0.5, [0], ("unknown", "low_confidence")),
    ],
)
def test_frozen_classification_boundaries(p, ids, expected):
    assert classify(p, ids) == expected


def projection(points, slot="slot"):
    return SlotProjection(
        CameraId("camera"),
        ParkingSlotId(slot),
        Polygon(tuple(Point(x, y) for x, y in points)),
    )


def test_assignment_best_margin_and_no_detection_is_not_free():
    a = projection([(0, 0), (0.4, 0), (0.4, 1), (0, 1)], "a")
    b = projection([(0.4, 0), (0.8, 0), (0.8, 1), (0.4, 1)], "b")
    detection = {"score": 0.60, "category": "car", "bbox": [1, 1, 3, 9]}
    assert assign_vehicles((a, b), 11, 11, [detection]) == {"a": [0], "b": []}
    detection["bbox"] = [2, 1, 6, 9]
    assert assign_vehicles((a, b), 11, 11, [detection]) == {"a": [], "b": []}
    detection["bbox"] = [1, 1, 3, 9]
    detection["score"] = 0.5999
    assert assign_vehicles((a, b), 11, 11, [detection]) == {"a": [], "b": []}
    small = projection([(0, 0), (0.15, 0), (0.15, 1), (0, 1)])
    detection.update(score=0.9, bbox=[0, 0, 10, 10])
    assert assign_vehicles((small,), 11, 11, [detection]) == {"slot": []}


@pytest.mark.parametrize(
    "box,expected",
    [
        ([0, 0, 4, 4], 10),
        ([2, 0, 4, 4], 4),
        ([0, 0, 1, 4], 4),
        ([2, 1.5, 4, 2.5], 0),
    ],
)
def test_concave_roi_intersection_with_disconnected_pieces(box, expected):
    # A C shape. The right-side clip leaves two disjoint strips.
    polygon = np.float32(
        [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)]
    )
    assert intersection_area(box, polygon) == pytest.approx(expected)


def test_asset_integrity_is_checked_before_model_loading(tmp_path):
    raw = tmp_path / "weights"
    raw.write_bytes(b"broken")
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "detector_weights": "weights",
                "per_slot_weights": "weights",
                "acpds_pooler": "weights",
            }
        )
    )
    with pytest.raises(ValueError, match="hash"):
        load_assets(assets)


def test_publish_is_exclusive_and_cleans_up_on_encoding_error(tmp_path):
    output = tmp_path / "observations.json"
    observe.save_result(output, {"prior": True})
    with pytest.raises(FileExistsError):
        observe.save_result(output, {"new": True})
    assert json.loads(output.read_text()) == {"prior": True}
    bad = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        observe.save_result(bad, {"bad": float("nan")})
    assert not bad.exists()
    assert list(tmp_path.iterdir()) == [output]


def test_publish_failure_leaves_no_partial_result(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("observe.os.link", fail)
    with pytest.raises(OSError):
        observe.save_result(output, {"complete": True})
    assert not list(tmp_path.iterdir())


def test_cli_from_another_directory_records_missing_time_and_never_overwrites(
    inputs, tmp_path
):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(Path(observe.__file__).resolve()),
        "--configuration",
        str(inputs.configuration),
        "--camera-id",
        "camera-east",
        "--image",
        str(inputs.image),
        "--session-id",
        "test",
        "--max-age-seconds",
        "60",
        "--view-confirmed",
        "--assets",
        str(inputs.assets),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, cwd="/tmp", text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    assert json.loads(raw)["status"] == "missing_capture_time"
    again = subprocess.run(command, cwd="/tmp", text=True, capture_output=True)
    assert again.returncode == 2
    assert output.read_bytes() == raw


def test_cli_rejects_naive_time(inputs, tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(Path(observe.__file__).resolve()),
            "--configuration",
            str(inputs.configuration),
            "--camera-id",
            "camera-east",
            "--image",
            str(inputs.image),
            "--session-id",
            "test",
            "--max-age-seconds",
            "60",
            "--assets",
            str(inputs.assets),
            "--captured-at",
            "2026-11-01T12:00:00",
            "--output",
            str(tmp_path / "out.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 2
    assert "UTC offset" in process.stderr
    assert not (tmp_path / "out.json").exists()

"""Operator registration manifests remain explicit and cannot bypass refusals."""

import json
from hashlib import sha256

import cv2
import numpy as np
import pytest

from parkvision.geometry_sequence_setup import propose_sequence


def _manifest(tmp_path, *, point_file=False):
    gray = np.random.default_rng(62).integers(20, 150, (480, 640), np.uint8)
    reference = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)
    source = cv2.warpAffine(reference, np.float32([[1, 0, 3], [0, 1, -2]]), (640, 480))
    controls = {
        "provenance": "synthetic manifest test; not an operator approval",
        "reference_pixel_sha256": sha256(reference.tobytes()).hexdigest(),
        "source_pixel_sha256": sha256(source.tobytes()).hexdigest(),
    }
    points = {
        "fit": [(80, 70), (320, 70), (550, 70), (80, 410), (320, 410), (550, 410)],
        "check": [
            (110, 120),
            (290, 120),
            (520, 120),
            (110, 350),
            (290, 350),
            (520, 350),
        ],
    }
    for role, coordinates in points.items():
        controls[role] = [
            {"id": f"{role}-{i}", "reference": [x, y], "source": [x + 3, y - 2]}
            for i, (x, y) in enumerate(coordinates)
        ]
    for name, image in [("reference.png", reference), ("source.png", source)]:
        assert cv2.imwrite(str(tmp_path / name), image)
    mask = np.full(reference.shape[:2], 255, np.uint8)
    assert cv2.imwrite(str(tmp_path / "background.png"), mask)
    point_path = tmp_path / "points.json"
    point_path.write_text(json.dumps(controls))
    spec = {
        "role": "controlled_synthetic_not_CCTV",
        "background_mask": "background.png",
        "frames": [
            {"image": "reference.png", "source_id": "synthetic", "camera_id": "fixed"},
            {
                "image": "source.png",
                "source_id": "synthetic",
                "camera_id": "fixed",
                "registration_points": "points.json" if point_file else controls,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path, spec, controls


def _propose(manifest, output):
    propose_sequence(
        manifest,
        output,
        parking_id="registration-test",
        parking_name="Test only",
        camera_id="fixed",
        camera_name="Synthetic fixed",
    )
    return json.loads((output / "evidence.json").read_bytes())


@pytest.mark.parametrize("point_file", [False, True])
def test_registration_points_keep_provenance_through_sequence_export(
    tmp_path, point_file
):
    manifest, _, controls = _manifest(tmp_path, point_file=point_file)
    source_bytes = (tmp_path / "points.json").read_bytes()
    output = tmp_path / "review"
    evidence = _propose(manifest, output)
    extra = evidence["multi_frame"]["frames"][1]
    assert extra["accepted"], extra
    assert extra["registration"]["method"] == "control_points"
    assert extra["registration"]["provenance"] == controls["provenance"]
    assert extra["registration"]["validation_count"] == 6
    assert (tmp_path / "points.json").read_bytes() == source_bytes
    assert all(c["status"] != "operator_verified" for c in evidence["candidates"])


def test_bad_operator_point_hash_does_not_fall_back_to_auto_registration(tmp_path):
    manifest, spec, controls = _manifest(tmp_path)
    controls["source_pixel_sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    evidence = _propose(manifest, tmp_path / "review")
    extra = evidence["multi_frame"]["frames"][1]
    assert not extra["accepted"]
    assert extra["registration"]["method"] == "control_points"
    assert extra["reason"] == "invalid_control_points"


def test_valid_operator_points_do_not_override_declared_camera_mismatch(tmp_path):
    manifest, spec, _ = _manifest(tmp_path)
    spec["frames"][1]["camera_id"] = "another-camera"
    manifest.write_text(json.dumps(spec))
    evidence = _propose(manifest, tmp_path / "review")
    extra = evidence["multi_frame"]["frames"][1]
    assert not extra["accepted"]
    assert extra["reason"] == "source_or_camera_mismatch"


@pytest.mark.parametrize("invalid", [None, [], 42])
def test_explicit_invalid_controls_cannot_silently_enable_automatic_registration(
    tmp_path, invalid
):
    manifest, spec, _ = _manifest(tmp_path)
    spec["frames"][1]["registration_points"] = invalid
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="registration_points"):
        _propose(manifest, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_per_frame_mask_overrides_the_global_mask_without_fallback(tmp_path):
    manifest, spec, _ = _manifest(tmp_path)
    spec["frames"][1].pop("registration_points")
    empty = np.zeros((480, 640), np.uint8)
    assert cv2.imwrite(str(tmp_path / "occluded-background.png"), empty)
    spec["frames"][1]["background_mask"] = "occluded-background.png"
    manifest.write_text(json.dumps(spec))
    evidence = _propose(manifest, tmp_path / "review")
    frame = evidence["multi_frame"]["frames"][1]
    assert not frame["accepted"]
    assert frame["registration_mask_sha256"] == sha256(empty.tobytes()).hexdigest()
    assert frame["reason"] == "insufficient_background_features"

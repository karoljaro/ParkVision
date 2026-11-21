"""Camera integrity protects fixed geometry without fitting on parked cars."""

import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

from parkvision.capture.camera_integrity import CameraIntegrityReference
from parkvision.configuration import load_configuration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/parking-setup-v1.json"
WIDTH, HEIGHT = 800, 560


def _texture():
    pixels = np.random.default_rng(31).integers(
        18, 230, (HEIGHT, WIDTH), dtype=np.uint8
    )
    return cv2.cvtColor(cv2.GaussianBlur(pixels, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)


def _artifact(directory, name, image):
    path = directory / name
    assert cv2.imwrite(str(path), image)
    return {"path": name, "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def configuration(tmp_path):
    data = json.loads(EXAMPLE.read_bytes())
    data["cameras"][0].update(image_width=WIDTH, image_height=HEIGHT)
    for projection in data["projections"]:
        if projection["camera_id"] == "camera-east":
            for point in projection["polygon"]:
                point["x"] = 0.15 + point["x"] * 0.9
    path = tmp_path / "approved.json"
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return load_configuration(path), sha256(raw).hexdigest()


def _manifest(tmp_path, configuration, image=None, masks=None):
    if image is None:
        image = _texture()
    if masks is None:
        y, x = np.indices((HEIGHT, WIDTH))
        check = (((x // 90 + y // 90) % 2) * 255).astype(np.uint8)
        masks = cv2.bitwise_not(check), check
    config, digest = configuration
    fit, check = masks
    data = {
        "schema_version": 1,
        "camera_id": "camera-east",
        "configuration_sha256": digest,
        "configuration_revision": config.revision,
        "image_size": [WIDTH, HEIGHT],
        "source": "controlled synthetic background, for testing",
        "approved_by": "synthetic-test-operator",
        "approved_at": "2026-11-21T09:00:00+01:00",
        "reference": _artifact(tmp_path, "ref.png", image),
        "fit_mask": _artifact(tmp_path, "fit.png", fit),
        "check_mask": _artifact(tmp_path, "check.png", check),
    }
    path = tmp_path / "view.json"
    path.write_text(json.dumps(data))
    return path


def _load(path, configuration):
    return CameraIntegrityReference.load(path, *configuration, "camera-east")


def test_native_pixels_verify_with_static_support_around_each_roi(
    tmp_path, configuration
):
    reference = _load(_manifest(tmp_path, configuration), configuration)
    result = reference.check(reference.image.copy())
    assert result.status == "verified", result
    assert result.reason == "native_roi_alignment_verified"
    assert result.diagnostics["independent_check_covers_roi"]
    assert result.diagnostics["fit_covers_roi"]
    assert result.diagnostics["max_roi_displacement_px"] < 0.1
    assert result.diagnostics["validation_p90_px"] < 0.1
    assert (
        result.reference_sha256
        == sha256((tmp_path / "ref.png").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "angle, scale, dx",
    [(0.0, 1.0, 5.0), (0.0, 1.016, 0.0), (1.4, 1.0, 0.0)],
    ids=["same-resolution-shift", "zoom", "rotation"],
)
def test_rejects_supported_change_with_same_image_dimensions(
    tmp_path, configuration, angle, scale, dx
):
    reference = _load(_manifest(tmp_path, configuration), configuration)
    transform = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, scale)
    transform[0, 2] += dx
    source = cv2.warpAffine(reference.image, transform, (WIDTH, HEIGHT))
    result = reference.check(source)
    assert result.status == "mismatch", result
    assert result.reason == "camera_view_changed"
    assert result.diagnostics["max_roi_displacement_px"] > 1.5
    assert source.shape == reference.image.shape


def test_missing_features_are_not_evidence_of_moved_camera(tmp_path, configuration):
    reference = _load(_manifest(tmp_path, configuration), configuration)
    result = reference.check(np.full_like(reference.image, 75))
    assert result.status == "unverifiable"
    assert result.reason.startswith("insufficient_")
    assert result.diagnostics["transform_source_to_reference"] is None


def test_independent_controls_must_spatially_support_roi(tmp_path, configuration):
    fit = np.zeros((HEIGHT, WIDTH), np.uint8)
    fit[:, :550] = 255
    check = np.zeros_like(fit)
    check[:, 600:] = 255
    reference = _load(
        _manifest(tmp_path, configuration, masks=(fit, check)), configuration
    )
    result = reference.check(reference.image.copy())
    assert result.status == "unverifiable", result
    assert result.reason == "insufficient_roi_spatial_support"
    assert result.diagnostics["registration_reason"] == "background_aligned"
    assert not result.diagnostics["independent_check_covers_roi"]


def test_vehicle_texture_inside_roi_cannot_prove_view(tmp_path, configuration):
    # Synthetic texture is visible exclusively inside a projected slot.
    empty = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    fit_image = empty.copy()
    fit_image[165:265, 202:255] = _texture()[165:265, 202:255]
    reference = _load(
        _manifest(tmp_path, configuration, image=fit_image), configuration
    )
    source = empty.copy()
    source[168:268, 207:260] = fit_image[165:265, 202:255]
    for mask in (reference.fit_mask, reference.check_mask):
        assert not mask[165:265, 202:255].any()
    result = reference.check(source)
    assert result.status == "unverifiable", result


def test_changed_pixels_at_slots_do_not_supply_evidence(tmp_path, configuration):
    reference = _load(_manifest(tmp_path, configuration), configuration)
    moved_car = reference.image.copy()
    moved_car[165:265, 202:255] = 50
    result = reference.check(moved_car)
    assert result.status == "verified", result
    assert result.diagnostics["max_roi_displacement_px"] < 0.1


@pytest.mark.parametrize(
    "field, wrong",
    [
        ("configuration_revision", 2),
        ("configuration_sha256", "f" * 64),
        ("camera_id", "camera-west"),
        ("image_size", [800, 559]),
        ("approved_by", ""),
        ("approved_at", "2026-11-21T09:00:00"),
    ],
)
def test_binding_and_explicit_approval_are_required(
    tmp_path, configuration, field, wrong
):
    path = _manifest(tmp_path, configuration)
    data = json.loads(path.read_bytes())
    data[field] = wrong
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _load(path, configuration)


@pytest.mark.parametrize("artifact", ["ref.png", "fit.png", "check.png"])
def test_tampered_assets_are_refused(tmp_path, configuration, artifact):
    path = _manifest(tmp_path, configuration)
    target = tmp_path / artifact
    target.write_bytes(target.read_bytes() + b"invalid")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load(path, configuration)


def test_reusing_overlapping_masks_is_refused(tmp_path, configuration):
    mask = np.full((HEIGHT, WIDTH), 255, np.uint8)
    path = _manifest(tmp_path, configuration, masks=(mask, mask))
    with pytest.raises(ValueError, match="disjoint"):
        _load(path, configuration)


def test_invalid_and_new_resolution_remain_distinct(tmp_path, configuration):
    reference = _load(_manifest(tmp_path, configuration), configuration)
    assert reference.check(np.empty((0, 0, 3), np.uint8)).status == "unverifiable"
    assert reference.check(reference.image[1:].copy()).status == "mismatch"


def test_moving_surface_in_fit_mask_conflicts_with_independent_landmarks(
    tmp_path, configuration
):
    reference_image = _texture()
    fit = np.zeros((HEIGHT, WIDTH), np.uint8)
    fit[:, 102:675] = 255
    check = np.zeros_like(fit)
    check[:, :82] = 255
    check[:, 693:] = 255
    reference = _load(
        _manifest(tmp_path, configuration, image=reference_image, masks=(fit, check)),
        configuration,
    )
    # Only a bounded foreground region moves. Independently selected static
    # landmarks explicitly contradict the resulting fit.
    source = reference_image.copy()
    source[:, 95:680] = reference_image[:, 88:673]
    result = reference.check(source)
    assert result.status == "unverifiable", result
    assert "held_out_alignment_failed" in result.diagnostics["registration_failures"]

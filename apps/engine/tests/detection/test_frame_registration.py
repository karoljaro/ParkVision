"""Phase 4C registration controls use fit/check evidence without ROI tuning."""

from copy import deepcopy
from hashlib import sha256

import cv2
import numpy as np
import pytest

from parkvision.detection.frame_registration import (
    _orb_matches,
    _refine,
    register_background,
)


def _texture(size=(640, 480), seed=23):
    width, height = size
    pixels = np.random.default_rng(seed).integers(
        20, 220, (height, width), dtype=np.uint8
    )
    return cv2.cvtColor(cv2.GaussianBlur(pixels, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)


def _shift(image, dx=4.25, dy=-2.5):
    height, width = image.shape[:2]
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))


def _controls(reference, source, dx=4.25, dy=-2.5):
    fit = [(80, 70), (320, 70), (550, 70), (80, 410), (320, 410), (550, 410)]
    check = [(110, 120), (290, 120), (520, 120), (110, 350), (290, 350), (520, 350)]
    return {
        "provenance": "controlled synthetic stable points; test only",
        "reference_pixel_sha256": sha256(reference.tobytes()).hexdigest(),
        "source_pixel_sha256": sha256(source.tobytes()).hexdigest(),
        "fit": [
            {"id": f"fit-{i}", "reference": [x, y], "source": [x + dx, y + dy]}
            for i, (x, y) in enumerate(fit)
        ],
        "check": [
            {"id": f"check-{i}", "reference": [x, y], "source": [x + dx, y + dy]}
            for i, (x, y) in enumerate(check)
        ],
    }


def _reject_controls(reference, source, controls):
    # Both an explicit input error and a diagnostic refusal are safe outcomes.
    try:
        result = register_background(reference, source, None, control_points=controls)
    except ValueError:
        return
    assert not result.accepted, result


@pytest.mark.parametrize("size", [(640, 480), (1920, 1440)])
def test_native_registration_corrects_small_subpixel_motion_with_holdout(size):
    reference = _texture(size)
    source = _shift(reference)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    result = register_background(reference, source, mask)
    assert result.accepted, result
    assert np.allclose(np.array(result.matrix)[:, 2], (-4.25, 2.5), atol=0.25)
    assert result.inliers >= 25
    assert result.residual_px <= 1.5
    assert result.validation_count > 0
    assert result.validation_p90_px <= 1.5
    assert result.validation_max_px <= 3
    assert result.correspondences


def test_auto_registration_handles_illumination_without_camera_motion():
    reference = _texture()
    source = cv2.convertScaleAbs(reference, alpha=0.8, beta=15)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    result = register_background(reference, source, mask)
    assert result.accepted, result
    assert np.allclose(np.array(result.matrix), [[1, 0, 0], [0, 1, 0]], atol=0.2)
    assert result.validation_p90_px <= 1.5


def test_local_refinement_keeps_brightness_shift_from_moving_the_edge():
    y, x = np.indices((240, 320), dtype=float)
    edge = 105 + 95 / (1 + np.exp(-(x - 155 + 0.5 * (y - 120)) / 3))
    noise = cv2.GaussianBlur(
        np.random.default_rng(13).normal(0, 1, edge.shape), (0, 0), 1.5
    )
    detail = 10 * np.exp(-((x - 164) ** 2 + (y - 132) ** 2) / 30)
    reference = np.uint8(np.clip(edge + 18 * noise + detail, 0, 255))
    darker = np.uint8(np.clip(reference.astype(np.int16) - 17, 0, 255))
    seed = np.float64([[155, 120]])

    # The legacy LK track drifted 3.34 px and passed its FB and NCC tests.
    source, target = _refine(reference, darker, seed, seed, 1.0)
    assert len(source) == 1
    np.testing.assert_allclose(source, target, atol=0.1)


def test_local_refinement_preserves_real_subpixel_motion_in_changed_light():
    reference = cv2.cvtColor(_texture(), cv2.COLOR_BGR2GRAY)
    motion = np.array([2.5, -1.75])
    frame = cv2.convertScaleAbs(_shift(reference, *motion), alpha=0.8, beta=12)
    reference_seeds = np.float64([[120, 140], [270, 280], [450, 380]])
    source_seeds = reference_seeds + [2, -1]

    source, target = _refine(reference, frame, source_seeds, reference_seeds, 1.0)
    assert len(source) >= 2
    assert np.max(np.linalg.norm(source - target - motion, axis=1)) < 0.12


def test_local_refinement_reuses_only_previously_evaluated_pairs(monkeypatch):
    from parkvision.detection import frame_registration

    reference = cv2.cvtColor(_texture(), cv2.COLOR_BGR2GRAY)
    seeds = np.float64([[120, 140], [4, 4]])
    cached = {}
    first = _refine(reference, reference, seeds, seeds, 1.0, local_matches=cached)
    assert len(first[0]) == 1
    assert len(cached) == 2
    assert sum(match is None for match in cached.values()) == 1

    local_match = frame_registration._refine_match
    seen = []

    def record_match(reference, image, seed, point, limit, radius, grid):
        seen.append(tuple(seed))
        return local_match(reference, image, seed, point, limit, radius, grid)

    monkeypatch.setattr(frame_registration, "_refine_match", record_match)
    updated = np.vstack((seeds[::-1], [[400, 260]]))
    repeated = _refine(
        reference, reference, updated, updated, 1.0, local_matches=cached
    )
    assert seen == [(400, 260)]
    for before, after in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(before[0], after[0])


@pytest.mark.parametrize("periodic", (False, True), ids=["edge", "repeated-corners"])
def test_local_refinement_rejects_equally_plausible_positions(periodic):
    y, x = np.indices((240, 320))
    if periodic:
        image = np.uint8(125 + 35 * np.cos(x * np.pi / 2) + 40 * np.cos(y * np.pi / 2))
    else:
        image = np.uint8(65 + 110 / (1 + np.exp(-(x - 154.5) / 3)))
    seed = np.float64([[155, 120]])

    source, target = _refine(image, image.copy(), seed, seed, 1.0)
    assert source.shape == target.shape == (0, 2)


def test_periodic_markings_do_not_prove_a_unique_correspondence():
    y, x = np.indices((480, 640))
    pattern = (((x // 24 + y // 24) % 2) * 190 + 25).astype(np.uint8)
    reference = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    source = _shift(reference, 24, 0)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    result = register_background(reference, source, mask)
    assert not result.accepted


def test_similar_window_details_are_matched_at_their_own_positions(monkeypatch):
    # The main feature in each window looks more like its counterpart in the
    # distant window. Each location also has a second descriptor for the ratio test.
    positions = np.float32([(64, 80), (72, 84), (208, 80), (216, 84)])
    reference_descriptors = np.vstack(
        [
            np.zeros(32, np.uint8),
            np.full(32, 255, np.uint8),
            np.array([255] + [0] * 31, np.uint8),
            np.full(32, 170, np.uint8),
        ]
    )
    source_descriptors = reference_descriptors.copy()
    source_descriptors[0, 0] = 254
    source_descriptors[2, 0] = 1
    shift = np.array([2, 1], dtype=np.float32)
    features = iter(
        [
            ([cv2.KeyPoint(*point, 7) for point in positions], reference_descriptors),
            (
                [cv2.KeyPoint(*point, 7) for point in positions + shift],
                source_descriptors,
            ),
        ]
    )

    class WindowFeatures:
        def detectAndCompute(self, image, mask):
            return next(features)

    monkeypatch.setattr(cv2, "ORB_create", lambda **kwargs: WindowFeatures())
    image = np.zeros((240, 320), dtype=np.uint8)
    mask = np.full_like(image, 255)
    source, reference, ratio = _orb_matches(image, image, mask)

    assert ratio == 1.0
    np.testing.assert_array_equal(reference, positions)
    np.testing.assert_array_equal(source, positions + shift)


def test_automatic_registration_can_diagnose_a_shift_outside_the_search_window():
    reference = _texture()
    source = _shift(reference, dx=80, dy=0)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    result = register_background(reference, source, mask)

    assert not result.accepted
    assert result.failure_reasons == ("camera_view_changed",)
    np.testing.assert_allclose(np.array(result.matrix)[:, 2], (-80, 0), atol=0.25)


def test_textured_moving_object_in_one_small_region_is_not_background_evidence():
    reference = np.full((480, 640, 3), 35, np.uint8)
    reference[180:300, 220:420] = _texture((200, 120))
    source = _shift(reference, 5, 0)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    assert not register_background(reference, source, mask).accepted


def test_bad_mask_dominated_by_moving_texture_fails_separate_static_check():
    reference = _texture()
    moved = _shift(reference, 7, 0)
    source = reference.copy()
    source[:, 110:530] = moved[:, 110:530]
    fit = np.zeros(reference.shape[:2], np.uint8)
    fit[:, 130:510] = 255
    check = np.zeros_like(fit)
    check[:, :90] = check[:, 550:] = 255
    result = register_background(reference, source, fit, validation_mask=check)
    assert not result.accepted
    assert result.inlier_ratio >= 0.6
    assert result.residual_px <= 1.5
    assert result.coverage >= 0.15 and result.occupied_cells >= 6
    assert result.validation_p90_px > 3
    assert result.reason == "held_out_alignment_failed"


def test_fit_and_validation_masks_cannot_reuse_the_same_pixels():
    reference = _texture()
    source = _shift(reference)
    mask = np.full(reference.shape[:2], 255, np.uint8)
    with pytest.raises(ValueError):
        register_background(reference, source, mask, validation_mask=mask.copy())


def test_operator_points_preserve_provenance_and_verify_unused_points():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    result = register_background(reference, source, None, control_points=controls)
    assert result.accepted, result
    assert np.allclose(np.array(result.matrix), [[1, 0, -4.25], [0, 1, 2.5]], atol=1e-9)
    assert result.method == "control_points"
    assert result.validation_count == 6
    assert result.validation_p90_px < 1e-8
    assert result.validation_max_px < 1e-8
    assert result.validation_coverage >= 0.15
    assert result.validation_cells >= 6
    assert len(result.correspondences) == 12
    assert controls == _controls(reference, source)


@pytest.mark.parametrize("field", ["source_pixel_sha256", "reference_pixel_sha256"])
def test_operator_points_cannot_be_reused_for_a_different_image(field):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls[field] = "0" * 64
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("field", ["fit", "check"])
def test_operator_points_need_enough_distributed_fit_and_check_points(field):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls[field] = controls[field][:4]
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("axis", [0, 1])
def test_operator_points_with_degenerate_collinear_geometry_are_refused(axis):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    for index, pair in enumerate(controls["fit"]):
        pair["reference"] = (
            [180, 60 + index * 70] if axis == 0 else [80 + index * 80, 200]
        )
        pair["source"] = [pair["reference"][0] + 4.25, pair["reference"][1] - 2.5]
    _reject_controls(reference, source, controls)


def test_points_concentrated_in_one_area_are_not_repaired_by_distant_check_points():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    for index, pair in enumerate(controls["fit"]):
        pair["reference"] = [240 + (index % 3) * 20, 200 + (index // 3) * 20]
        pair["source"] = [pair["reference"][0] + 4.25, pair["reference"][1] - 2.5]
    _reject_controls(reference, source, controls)


def test_wide_fit_cannot_hide_check_points_concentrated_in_one_area():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    for index, pair in enumerate(controls["check"]):
        pair["reference"] = [240 + (index % 3) * 20, 200 + (index // 3) * 20]
        pair["source"] = [pair["reference"][0] + 4.25, pair["reference"][1] - 2.5]
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("field", ["fit", "check"])
def test_duplicate_operator_correspondences_are_refused(field):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls[field][1] = deepcopy(controls[field][0])
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("field", ["fit", "check"])
def test_duplicate_coordinates_do_not_gain_support_under_a_fresh_id(field):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    duplicate = deepcopy(controls[field][0])
    duplicate["id"] = "new-name-for-an-existing-point"
    controls[field].append(duplicate)
    _reject_controls(reference, source, controls)


def test_same_point_cannot_count_as_both_fit_and_validation():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls["check"][0] = deepcopy(controls["fit"][0])
    controls["check"][0]["id"] = "different-name-for-same-point"
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 650])
def test_invalid_original_pixel_coordinates_are_refused(value):
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls["fit"][0]["source"][0] = value
    _reject_controls(reference, source, controls)


def test_missing_operator_point_provenance_is_not_implicitly_accepted():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls["provenance"] = ""
    _reject_controls(reference, source, controls)


def test_manual_fit_outlier_is_not_silently_discarded_by_ransac():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    controls["fit"][-1]["source"][0] += 30
    _reject_controls(reference, source, controls)


def test_good_fit_cannot_hide_contradicting_unused_check_points():
    reference = _texture()
    source = _shift(reference)
    controls = _controls(reference, source)
    for pair in controls["check"]:
        pair["source"][0] += 7
    _reject_controls(reference, source, controls)


def test_consistent_coordinates_do_not_validate_unrelated_image_pixels():
    reference = _texture()
    source = _texture(seed=42)
    controls = _controls(reference, source, 0, 0)
    _reject_controls(reference, source, controls)


def test_manual_coordinates_on_textureless_images_are_not_observable_evidence():
    reference = np.full((480, 640, 3), 80, np.uint8)
    source = reference.copy()
    controls = _controls(reference, source, 0, 0)
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("dx,dy", [(35, 0), (0, 35)])
def test_consistent_operator_points_do_not_override_stationary_camera_limits(dx, dy):
    reference = _texture()
    source = _shift(reference, dx, dy)
    controls = _controls(reference, source, dx, dy)
    _reject_controls(reference, source, controls)


@pytest.mark.parametrize("angle,scale", [(3, 1), (0, 1.025)])
def test_operator_points_do_not_allow_rotation_or_zoom_beyond_limits(angle, scale):
    reference = _texture()
    matrix = cv2.getRotationMatrix2D((320, 240), angle, scale)
    source = cv2.warpAffine(reference, matrix, (640, 480))
    controls = _controls(reference, source, 0, 0)
    for role in ("fit", "check"):
        for pair in controls[role]:
            pair["source"] = (
                matrix[:, :2] @ np.array(pair["reference"]) + matrix[:, 2]
            ).tolist()
    _reject_controls(reference, source, controls)


def test_operator_points_support_a_small_genuine_similarity_transform():
    reference = _texture()
    forward = cv2.getRotationMatrix2D((320, 240), -0.2, 1.002)
    source = cv2.warpAffine(reference, forward, (640, 480))
    controls = _controls(reference, source, 0, 0)
    for role in ("fit", "check"):
        for pair in controls[role]:
            pair["source"] = (
                forward[:, :2] @ np.array(pair["reference"]) + forward[:, 2]
            ).tolist()
    result = register_background(reference, source, None, control_points=controls)
    assert result.accepted, result
    assert np.allclose(result.matrix, cv2.invertAffineTransform(forward), atol=1e-8)


def test_fit_and_holdout_do_not_reuse_orb_scale_duplicates_of_one_corner():
    reference = _texture()
    source = _shift(reference)
    result = register_background(
        reference, source, np.full(reference.shape[:2], 255, np.uint8)
    )
    assert result.accepted
    fit = [p for p in result.correspondences if p["role"] == "fit"]
    check = [p for p in result.correspondences if p["role"] == "check"]
    for coordinates in ("source", "reference"):
        a = np.array([p[coordinates] for p in fit])
        b = np.array([p[coordinates] for p in check])
        assert np.linalg.norm(a[:, None] - b[None, :], axis=2).min() >= 12 - 1e-6


def test_static_border_regions_are_not_discarded_by_resized_orb_margin():
    reference = _texture()
    source = _shift(reference, 2, 0)
    mask = np.zeros(reference.shape[:2], np.uint8)
    mask[18:28] = mask[452:462] = 255
    mask[:, 18:28] = mask[:, 612:622] = 255
    result = register_background(reference, source, mask)
    assert result.accepted
    assert result.occupied_cells >= 6 and result.validation_cells >= 6


def test_accepted_lighting_frame_with_no_new_geometry_preserves_reference_drafts():
    from test_multi_frame import _GeometryDetector, _input, _row
    from parkvision.detection.multi_frame import analyze_frames

    reference = _texture()
    source = cv2.add(reference, np.full_like(reference, 20))
    baseline = _row().result()
    result = analyze_frames(
        (_input(reference), _input(source, 1)),
        background_mask=np.full(reference.shape[:2], 255, np.uint8),
        detector=_GeometryDetector((baseline, baseline)),
    )
    assert result.frames[1]["accepted"]
    assert result.frames[1]["registration"]["validation_count"] >= 12
    assert tuple(
        c.proposal.polygon for c in result.geometry.candidates if c.proposal
    ) == tuple(c.proposal.polygon for c in baseline.candidates if c.proposal)

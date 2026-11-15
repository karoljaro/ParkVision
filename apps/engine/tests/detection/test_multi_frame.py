"""Multi-frame safety contracts with controlled geometry and background motion."""

from dataclasses import replace
from datetime import UTC, datetime
from math import hypot

import cv2
import numpy as np
import pytest

from parkvision.capture.frame import Frame
from parkvision.detection.frame_registration import (
    IDENTITY,
    Registration,
    register_background,
)
from parkvision.detection.initialization import GeometryInitialization
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.multi_frame import GeometryFrame, analyze_frames
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector
from parkvision.detection.segment_group import SegmentGroup


def _texture(seed: int = 5) -> np.ndarray:
    random = np.random.default_rng(seed)
    image = random.integers(20, 210, (480, 640), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _input(image: np.ndarray, index: int = 0, **kwargs) -> GeometryFrame:
    return GeometryFrame(
        Frame(kwargs.get("camera", "fixed-01"), index, datetime.now(UTC), image),
        kwargs.get("source", "diagnostic-source"),
        kwargs.get("image_id", f"frame-{index}"),
    )


class _GeometryDetector(OpenCVSlotDetector):
    def __init__(self, scenes):
        super().__init__()
        self.scenes = scenes
        self.calls = []

    def analyze(self, frame):
        self.calls.append(frame.sequence)
        return self.scenes[frame.sequence]


class _Scene:
    def __init__(self):
        self.pairs = []
        self.sources = {}
        self.context = []

    def paint(self, start, end, width=4.0):
        axis_id = len(self.pairs)
        axis = LineSegment(axis_id, start, end)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        edges = []
        for side in (-0.5, 0.5):
            offset = (-dy * width * side / length, dx * width * side / length)
            edge_id = 1000 + 2 * axis_id + len(edges)
            edge = LineSegment(
                edge_id,
                (start[0] + offset[0], start[1] + offset[1]),
                (end[0] + offset[0], end[1] + offset[1]),
            )
            edges.append(edge)
            self.sources[edge_id] = SegmentGroup(
                (LineSegment(10000 + edge_id, edge.start, edge.end),)
            )
        self.pairs.append(MarkingPair(tuple(edges), axis, (width, width), 1.0))

    def result(self) -> GeometryInitialization:
        segments = tuple(
            line for group in self.sources.values() for line in group.segments
        )
        fitted = tuple(edge for pair in self.pairs for edge in pair.edges)
        return OpenCVSlotDetector().assemble_evidence(
            _input(_texture()).frame,
            segments,
            fitted,
            self.sources,
            tuple(self.pairs),
        )


def _row(indices=(0, 1, 2, 3), *, rear=True, perspective=False):
    scene = _Scene()
    scene.paint((50.0, 80.0), (580.0, 80.0))
    if rear:
        scene.paint((40.0, 350.0), (600.0, 350.0))
    for index in indices:
        x = (100.0, 230.0, 380.0, 530.0)[index]
        rear_x = (80.0, 220.0, 390.0, 555.0)[index] if perspective else x
        scene.paint((x, 80.0), (rear_x, 350.0))
    return scene


@pytest.fixture
def trusted_registration(monkeypatch):
    """Geometry unit tests isolate fusion from independently tested registration."""
    monkeypatch.setattr(
        "parkvision.detection.multi_frame.register_background",
        lambda *args: Registration(True, "controlled_test_alignment", IDENTITY),
    )


def _fuse(first, second):
    image = _texture()
    changed = image.copy()
    changed[120:280, 200:440] = 20
    return analyze_frames(
        (_input(image), _input(changed, 1)),
        detector=_GeometryDetector((first.result(), second.result())),
    )


def test_identical_images_preserve_geometry_and_do_not_count_support_twice():
    image = _texture()
    baseline = _row().result()
    detector = _GeometryDetector((baseline,))
    result = analyze_frames((_input(image), _input(image.copy(), 1)), detector=detector)
    assert result.geometry is baseline
    assert detector.calls == [0]
    assert result.frames[1]["reason"] == "duplicate_pixels"
    assert all(marking["support_kind"] == "single_frame" for marking in result.markings)
    assert len(result.observations) == len(baseline.pairs)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"camera": "different-camera"}, "source_or_camera_mismatch"),
        ({"source": "different-location"}, "source_or_camera_mismatch"),
        ({"resize": True}, "resolution_mismatch"),
    ],
)
def test_incompatible_input_is_rejected_before_analysis(changes, reason):
    image = _texture()
    second = image[:240] if changes.get("resize") else image
    detector = _GeometryDetector((_row().result(),))
    result = analyze_frames(
        (_input(image), _input(second, 1, **changes)), detector=detector
    )
    assert detector.calls == [0]
    assert not result.frames[1]["accepted"]
    assert result.frames[1]["reason"] == reason


def test_nonidentical_images_need_a_declared_stationary_background():
    result = analyze_frames(
        (_input(_texture()), _input(_texture(8), 1)),
        detector=_GeometryDetector((_row().result(),)),
    )
    assert result.frames[1]["reason"] == "stationary_background_mask_required"


def test_near_duplicate_cannot_create_multiple_frame_support(trusted_registration):
    image = _texture()
    near = cv2.add(image, np.ones_like(image))
    result = analyze_frames(
        (_input(image), _input(near, 1)),
        detector=_GeometryDetector((_row().result(),)),
    )
    assert result.frames[1]["reason"] == "near_duplicate_no_new_evidence"
    assert all(item["support_kind"] == "single_frame" for item in result.markings)


def test_small_new_corner_is_not_discarded_by_global_image_similarity(
    trusted_registration,
):
    image = _texture()
    changed = image.copy()
    changed[80:87, 100:107] = 230
    result = analyze_frames(
        (_input(image), _input(changed, 1)),
        detector=_GeometryDetector((_row(rear=False).result(), _row().result())),
    )
    assert result.frames[1]["accepted"]
    assert len(result.geometry.pairs) == 6


def test_complementary_separators_recover_unique_geometry_with_provenance(
    trusted_registration,
):
    first, second = _row((0, 1, 2)), _row((1, 2, 3))
    result = _fuse(first, second)
    assert all(frame["accepted"] for frame in result.frames)
    assert len(result.geometry.pairs) == 6
    assert len(result.markings) == 6
    assert (
        sum(item["support_kind"] == "multiple_frames" for item in result.markings) == 4
    )
    assert len(result.observations) == 10
    assert any(
        candidate.proposal is not None for candidate in result.geometry.candidates
    )
    for observation in result.observations:
        assert observation["source_lsd"]
        assert observation["reference_lsd_ids"]
        assert observation["source_axis"]["start"] == tuple(
            observation["reference_axis"]["start"]
        )


def test_missing_rail_and_corners_can_be_observed_in_second_frame(
    trusted_registration,
):
    first = _row(rear=False)
    second = _row()
    baseline = first.result()
    result = _fuse(first, second)
    assert len(result.geometry.topology.junctions) > len(baseline.topology.junctions)
    assert len(result.geometry.pairs) == 6
    assert any(
        candidate.proposal is not None for candidate in result.geometry.candidates
    )


def test_variable_perspective_widths_are_not_forced_equal(trusted_registration):
    result = _fuse(_row((0, 1, 2), perspective=True), _row((1, 2, 3), perspective=True))
    assert len(result.geometry.pairs) == 6
    assert sum(c.proposal is not None for c in result.geometry.candidates) >= 2


def test_open_t_row_remains_partial_without_physical_depth(trusted_registration):
    first, second = _Scene(), _Scene()
    for scene, indices in ((first, (0, 1, 2)), (second, (1, 2, 3))):
        for index in indices:
            x, y = (100 + 125 * index, 80 + 10 * index)
            scene.paint((float(x), float(y)), (float(x + 5), float(340 + 4 * index)))
            scene.paint((float(x - 18), float(y)), (float(x + 18), float(y)))
    result = _fuse(first, second)
    partial = [c for c in result.geometry.candidates if c.geometry_kind == "partial"]
    assert partial
    assert all(c.proposal is None and c.status == "unresolved" for c in partial)
    assert all("physical_depth" in c.missing_boundaries for c in partial)
    assert not any(c.proposal is not None for c in result.geometry.candidates)


def test_displaced_stripe_is_recorded_as_conflict_not_double_support(
    trusted_registration,
):
    first, second = _Scene(), _Scene()
    first.paint((100.0, 80.0), (100.0, 350.0))
    second.paint((106.0, 80.0), (106.0, 350.0))
    result = _fuse(first, second)
    assert len(result.geometry.pairs) == 1
    assert len(result.conflicts) == 1
    assert result.conflicts[0]["reason"] == "inconsistent_stripe_position_or_width"
    assert result.markings[0]["frame_indices"] == [0]


def test_distinct_parallel_separators_are_not_merged_by_direction(
    trusted_registration,
):
    first, second = _Scene(), _Scene()
    first.paint((100.0, 80.0), (100.0, 350.0))
    second.paint((230.0, 80.0), (230.0, 350.0))
    result = _fuse(first, second)
    assert len(result.geometry.pairs) == 2
    assert not result.conflicts
    assert all(item["support_kind"] == "single_frame" for item in result.markings)


def test_source_coordinates_survive_small_alignment(monkeypatch):
    first, shifted = _row(), _Scene()
    for pair in first.pairs:
        start, end = pair.centerline.start, pair.centerline.end
        shifted.paint((start[0] + 3, start[1] - 2), (end[0] + 3, end[1] - 2))
    matrix = ((1.0, 0.0, -3.0), (0.0, 1.0, 2.0))
    monkeypatch.setattr(
        "parkvision.detection.multi_frame.register_background",
        lambda *args: Registration(True, "controlled_test_alignment", matrix),
    )
    result = _fuse(first, shifted)
    assert len(result.geometry.pairs) == len(first.pairs)
    for observation in result.observations:
        if observation["frame_index"] == 1:
            source = np.array(observation["source_axis"]["start"])
            target = np.array(observation["reference_axis"]["start"])
            assert np.allclose(target, source + (-3, 2))
    assert result.frames[1]["transform"] == matrix


def test_short_hidden_rail_span_is_covered_only_by_real_second_frame_fragments(
    trusted_registration,
):
    first, second = _row(), _row()
    for edge in first.pairs[0].edges:
        first.sources[edge.segment_id] = SegmentGroup(
            (
                LineSegment(
                    20000 + edge.segment_id * 2, edge.start, (222.0, edge.start[1])
                ),
                LineSegment(
                    20001 + edge.segment_id * 2, (238.0, edge.start[1]), edge.end
                ),
            )
        )
    result = _fuse(first, second)
    assert len(result.geometry.pairs) == 6
    source_groups = dict(result.geometry.source_groups)
    for edge in result.geometry.pairs[0].edges:
        group = source_groups[edge.segment_id]
        assert len(group.segments) == 3
        # The continuous support is supplied by frame 2, not a fabricated bridge.
        assert sorted(round(s.length) for s in group.segments) == [172, 342, 530]
    assert any(c.proposal is not None for c in result.geometry.candidates)


def test_repeated_unseen_span_is_not_filled_by_the_fitted_envelope(
    trusted_registration,
):
    first, second = _Scene(), _Scene()
    for scene in (first, second):
        scene.paint((100.0, 80.0), (100.0, 350.0))
        for edge in scene.pairs[0].edges:
            scene.sources[edge.segment_id] = SegmentGroup(
                (
                    LineSegment(
                        21000 + edge.segment_id * 2, edge.start, (edge.start[0], 130.0)
                    ),
                    LineSegment(
                        21001 + edge.segment_id * 2, (edge.start[0], 300.0), edge.end
                    ),
                )
            )
    result = _fuse(first, second)
    assert len(result.geometry.pairs) == 1
    assert not any(c.proposal is not None for c in result.geometry.candidates)
    groups = dict(result.geometry.source_groups)
    for edge in result.geometry.pairs[0].edges:
        fragments = groups[edge.segment_id].segments
        assert all(s.start[1] >= 300 or s.end[1] <= 130 for s in fragments)


def test_repeated_hatching_with_perimeters_never_produces_a_full_roi(
    trusted_registration,
):
    scenes = (_Scene(), _Scene())
    for scene in scenes:
        for y in (75.0, 115.0, 155.0):
            scene.paint((55.0, y), (140.0, y + 50.0))
        scene.paint((55.0, 45.0), (55.0, 260.0))
        scene.paint((155.0, 45.0), (155.0, 280.0))
    result = _fuse(*scenes)
    assert not any(c.proposal is not None for c in result.geometry.candidates)


def test_new_frame_cannot_remove_or_modify_reference_full_drafts(
    trusted_registration,
):
    first = _row()
    second = _Scene()
    second.paint((106.0, 80.0), (106.0, 350.0))
    baseline = [c.proposal for c in first.result().candidates if c.proposal is not None]
    assert baseline
    result = _fuse(first, second)
    updated = [c.proposal for c in result.geometry.candidates if c.proposal is not None]
    assert all(proposal in updated for proposal in baseline)
    assert all(
        candidate.status != "operator_verified"
        for candidate in result.geometry.candidates
    )


def test_registration_aligns_small_shift_using_distributed_background():
    reference = _texture()
    image = cv2.warpAffine(reference, np.float32([[1, 0, 4], [0, 1, -3]]), (640, 480))
    result = register_background(reference, image, np.full((480, 640), 255, np.uint8))
    assert result.accepted, result
    assert result.inliers >= 25
    assert result.coverage >= 0.15
    assert result.residual_px < 1.5
    assert np.allclose(np.array(result.matrix)[:, 2], (-4, 3), atol=0.3)


@pytest.mark.parametrize(
    "mode", ["large_shift", "zoom", "rotation", "different_camera"]
)
def test_registration_rejects_changed_view_despite_equal_dimensions(mode):
    reference = _texture()
    if mode == "large_shift":
        matrix = np.float32([[1, 0, 35], [0, 1, 0]])
    elif mode == "zoom":
        matrix = cv2.getRotationMatrix2D((320, 240), 0, 1.025)
    else:
        matrix = cv2.getRotationMatrix2D((320, 240), 3, 1)
    image = (
        _texture(22)
        if mode == "different_camera"
        else cv2.warpAffine(reference, matrix, (640, 480))
    )
    result = register_background(reference, image, np.full((480, 640), 255, np.uint8))
    assert not result.accepted
    assert result.reason in {
        "camera_view_changed",
        "insufficient_mutual_matches",
        "unreliable_background_fit",
    }


def test_registration_accepts_lighting_change_without_inventing_motion():
    reference = _texture()
    image = cv2.convertScaleAbs(reference, alpha=0.8, beta=20)
    result = register_background(reference, image, np.full((480, 640), 255, np.uint8))
    assert result.accepted, result
    assert np.allclose(np.array(result.matrix), np.array(IDENTITY), atol=0.15)


def test_registration_rejects_unobservable_or_localized_background():
    image = np.full((480, 640, 3), 30, np.uint8)
    assert (
        register_background(image, image, np.full((480, 640), 255, np.uint8)).reason
        == "insufficient_background_features"
    )
    image = _texture()
    mask = np.zeros((480, 640), np.uint8)
    mask[80:180, 80:220] = 255
    result = register_background(image, image, mask)
    assert not result.accepted
    assert result.reason == "insufficient_spatial_coverage"


def test_bad_background_mask_is_explicit_input_error():
    with pytest.raises(ValueError, match="Background mask"):
        register_background(_texture(), _texture(), np.ones((10, 10), np.uint8))


def test_empty_or_duplicate_frame_identifiers_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        analyze_frames(())
    frame = _input(_texture())
    with pytest.raises(ValueError, match="unique"):
        analyze_frames((frame, frame))
    with pytest.raises(ValueError, match="nonempty"):
        analyze_frames((replace(frame, source_id=""),))


@pytest.mark.parametrize("covered", [True, False])
def test_new_stripe_can_join_old_fragments_only_with_source_coverage(
    trusted_registration, covered
):
    first = _Scene()
    first.paint((50.0, 80.0), (295.0, 80.0))
    first.paint((345.0, 80.0), (580.0, 80.0))
    first.paint((40.0, 350.0), (600.0, 350.0))
    for x in (100.0, 230.0, 380.0, 530.0):
        first.paint((x, 80.0), (x, 350.0))
    second = _row()
    if not covered:
        for edge in second.pairs[0].edges:
            raw = second.sources[edge.segment_id].segments[0]
            second.sources[edge.segment_id] = SegmentGroup(
                (
                    replace(raw, end=(295.0, raw.end[1])),
                    replace(
                        raw,
                        segment_id=raw.segment_id + 100000,
                        start=(345.0, raw.start[1]),
                    ),
                )
            )
    result = _fuse(first, second)
    assert len(result.geometry.pairs) == (6 if covered else 7)
    reasons = [item["reason"] for item in result.conflicts]
    assert ("ambiguous_or_same_frame_match" in reasons) is not covered


def test_new_internal_separator_disputes_an_unapproved_reference_draft(
    trusted_registration,
):
    first, second = _row(), _row()
    baseline = first.result()
    second.paint((165.0, 80.0), (165.0, 350.0))
    result = _fuse(first, second)
    assert any(
        c["reason"] == "reference_draft_has_new_internal_separator"
        for c in result.conflicts
    )
    originals = [c for c in baseline.candidates if c.proposal is not None]
    remaining = [
        c.proposal.polygon for c in result.geometry.candidates if c.proposal is not None
    ]
    assert any(c.proposal.polygon not in remaining for c in originals)
    assert any(
        c.status == "unresolved" and "internal separator" in c.reason
        for c in result.geometry.candidates
    )


def test_partial_updates_close_only_after_a_real_rear_boundary_is_observed(
    trusted_registration,
):
    first = _Scene()
    first.paint((50.0, 80.0), (580.0, 80.0))
    for x, tip in zip(
        (100.0, 230.0, 380.0, 530.0), (280.0, 330.0, 300.0, 350.0), strict=True
    ):
        first.paint((x, 80.0), (x, tip))
    baseline = first.result()
    assert any(c.geometry_kind == "partial" for c in baseline.candidates)
    result = _fuse(first, _row())
    assert result.hypothesis_updates
    assert any(
        item["state"] == "closed_with_source_support"
        for item in result.hypothesis_updates
    )
    assert any(c.proposal is not None for c in result.geometry.candidates)

"""A connected hatched perimeter is negative evidence, not an unknown bay depth."""

import cv2
import numpy as np
import pytest
from test_open_rows import _Scene, _assert_partial_only, _discrete_t_row
from test_multi_frame import _GeometryDetector, _input, _texture
from parkvision.detection.multi_frame import analyze_frames
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector


def _hatched_corner(*, curb=False, repetitions=4):
    scene = _Scene()
    paint = scene.paint
    paint((80.0, 80.0), (360.0, 108.0))
    (scene.boundary if curb else paint)((80.0, 80.0), (35.0, 300.0))
    strokes = (
        ((70.0, 125.0), (230.0, 300.0)),
        ((150.0, 87.0), (340.0, 300.0)),
        ((230.0, 95.0), (425.0, 300.0)),
        ((310.0, 103.0), (435.0, 235.0)),
    )
    for start, end in strokes[:repetitions]:
        scene.paint(start, end)
    return scene


def _result(scene):
    image = _texture()
    return OpenCVSlotDetector().assemble_evidence(
        _input(image).frame,
        tuple(s for group in scene.sources.values() for s in group.segments),
        tuple(e for p in scene.pairs for e in p.edges) + tuple(scene.context),
        scene.sources,
        tuple(scene.pairs),
    )


@pytest.mark.parametrize("curb", [False, True])
def test_repeated_diagonal_strokes_on_two_perimeter_arms_are_rejected(curb):
    scene = _hatched_corner(curb=curb)
    rows, candidates = scene.reconstruct()
    if not curb:
        assert candidates, "Exercise an actual partial, not an empty scene"
    # Unclassified curb context can already withhold the row before validation.
    assert not any(c.rejection_code is None for c in candidates)
    assert all(c.rejection_code == "non_parking_hatching" for c in candidates)
    assert all(c.rejection_line_ids and c.proposal is None for c in candidates)
    assert all(row.rejection_code == "non_parking_hatching" for row in rows)
    assert all(not any(row.compatible_pairs) for row in rows)


def test_two_strokes_do_not_prove_a_nonparking_pattern():
    _, candidates = _hatched_corner(repetitions=2).reconstruct()
    assert candidates
    assert all(c.rejection_code is None for c in candidates)


def test_angled_parking_row_on_one_front_boundary_stays_unresolved():
    scene = _Scene()
    scene.paint((40.0, 60.0), (380.0, 60.0))
    for x in (65.0, 145.0, 225.0, 305.0):
        scene.paint((x, 60.0), (x + 90.0, 245.0))
        scene.paint((x - 15.0, 60.0), (x + 15.0, 60.0))
    result = _result(scene)
    assert any(c.proposal is not None for c in result.candidates)
    assert all(c.rejection_code is None for c in result.candidates)


def test_disjoint_t_caps_and_partial_visibility_are_not_hatching():
    scene, _ = _discrete_t_row()
    _, candidates = scene.reconstruct()
    _assert_partial_only(candidates)
    assert all(c.rejection_code is None for c in candidates)


def test_known_hatched_pattern_does_not_gain_slot_support_from_additional_frames():
    scene = _hatched_corner()
    baseline = _result(scene)
    assert any(c.rejection_code for c in baseline.candidates)
    image = _texture()
    changed = cv2.add(image, np.full_like(image, 20))
    result = analyze_frames(
        (_input(image), _input(changed, 1)),
        background_mask=np.full(image.shape[:2], 255, np.uint8),
        detector=_GeometryDetector((baseline, baseline)),
    )
    assert result.frames[1]["accepted"]
    assert not any(c.proposal for c in result.geometry.candidates)
    partial = [c for c in result.geometry.candidates if c.geometry_kind == "partial"]
    assert partial and all(c.rejection_code == "non_parking_hatching" for c in partial)
    assert all(u["state"] == "rejected_non_parking" for u in result.hypothesis_updates)


def test_hatched_zone_next_to_valid_open_bays_does_not_reject_the_valid_row():
    scene = _hatched_corner()
    stems = []
    for x, tip in zip((470.0, 530.0, 590.0), (180.0, 230.0, 210.0), strict=True):
        stems.append(scene.paint((x, 80.0), (x, tip)))
        scene.paint((x - 16.0, 80.0), (x + 16.0, 80.0))
    result = _result(scene)
    active = [
        c
        for c in result.candidates
        if c.geometry_kind == "partial" and c.rejection_code is None
    ]
    rejected = [
        c for c in result.candidates if c.rejection_code == "non_parking_hatching"
    ]
    assert rejected and active
    assert all(set(c.separator_ids) <= set(stems) for c in active)


def test_partially_observed_t_stems_remain_active_without_a_physical_depth():
    from dataclasses import replace
    from math import dist

    scene, stems = _discrete_t_row()
    for axis in [p for p in scene.pairs if p.centerline.segment_id in stems]:
        for edge in axis.edges:
            group = scene.sources[edge.segment_id]
            scene.sources[edge.segment_id] = replace(
                group,
                segments=tuple(
                    replace(
                        s, end=tuple(a + 0.6 * (b - a) for a, b in zip(s.start, s.end))
                    )
                    for s in group.segments
                ),
            )
    _, candidates = scene.reconstruct()
    _assert_partial_only(candidates)
    assert all(c.missing_boundaries and c.rejection_code is None for c in candidates)
    lengths = {p.centerline.segment_id: p.centerline.length for p in scene.pairs}
    assert all(
        dist(b.start, b.end) < 0.8 * lengths[b.line_ids[0]]
        for c in candidates
        for b in c.boundaries
    )


def test_real_p9_geometry_excerpt_is_rejected_with_perimeter_witnesses():
    import json
    from pathlib import Path
    from datetime import UTC, datetime
    from parkvision.capture.frame import Frame
    from parkvision.detection.line_segment import LineSegment
    from parkvision.detection.marking_pair import MarkingPair
    from parkvision.detection.segment_group import SegmentGroup

    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/partial-hatched-corner.json").read_text()
    )

    def line(d):
        return LineSegment(d["segment_id"], tuple(d["start"]), tuple(d["end"]))

    raw = {s["segment_id"]: line(s) for s in data["segments"]}
    pairs = tuple(
        MarkingPair(
            tuple(line(e) for e in p["edges"]),
            line(p["centerline"]),
            tuple(p["widths_px"]),
            p["overlap_ratio"],
        )
        for p in data["pairs"]
    )
    groups = {
        g["fitted_line_id"]: SegmentGroup(tuple(raw[i] for i in g["segment_ids"]))
        for g in data["source_groups"]
    }
    width, height = data["size"]
    frame = Frame(
        "diagnostic", 0, datetime.now(UTC), np.zeros((height, width, 3), np.uint8)
    )
    result = OpenCVSlotDetector().assemble_evidence(
        frame,
        tuple(raw.values()),
        tuple(e for p in pairs for e in p.edges),
        groups,
        pairs,
    )
    candidate = next(
        c
        for c in result.candidates
        if set(c.separator_ids) == {5620, 5629} and c.geometry_kind == "partial"
    )
    assert candidate.rejection_code == "non_parking_hatching"
    assert set(candidate.rejection_line_ids[:2]) == {5614, 5626}
    assert len(candidate.rejection_line_ids[2:]) >= 3
    assert candidate.status == "unresolved" and candidate.proposal is None

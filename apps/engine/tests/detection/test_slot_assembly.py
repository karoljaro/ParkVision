from math import dist

import pytest

from parkvision.detection.initialization import SlotCandidate
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_topology import find_topology_candidates
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import PixelPoint
from parkvision.detection.slot_assembly import assemble_slots
from parkvision.detection.topology_support_measurement import measure_topology_support
from parkvision.domain.geometry.point import Point
from parkvision.domain.types import CameraId


def _row(
    *,
    rear: bool = True,
    separator_count: int = 4,
    shortened_tip: float = 190,
    interruption: tuple[float, float] | None = None,
    continuation: bool = False,
) -> tuple[SlotCandidate, ...]:
    """Separate exact geometric evidence from variability in OpenCV extraction."""
    pairs = []
    groups = {}

    def marking(start: PixelPoint, end: PixelPoint, gap: tuple[float, float] | None):
        axis = LineSegment(len(pairs), start, end)
        edges = []
        for side in (-2.0, 2.0):
            dx, dy = (side, 0.0) if start[0] == end[0] else (0.0, side)
            edge = LineSegment(
                100 + 2 * len(pairs) + len(edges),
                (start[0] + dx, start[1] + dy),
                (end[0] + dx, end[1] + dy),
            )
            edges.append(edge)
            source_start = edge.segment_id * 10
            if gap is None:
                fragments = (LineSegment(source_start, edge.start, edge.end),)
            else:
                lo, hi = gap
                fragments = (
                    LineSegment(source_start, edge.start, (edge.start[0], lo)),
                    LineSegment(source_start + 1, (edge.start[0], hi), edge.end),
                )
            groups[edge.segment_id] = SegmentGroup(fragments)
        pairs.append(MarkingPair((edges[0], edges[1]), axis, (4.0, 4.0), 1.0))

    for index, x in enumerate((65.0, 125.0, 195.0, 275.0)[:separator_count]):
        tip = shortened_tip if index == 1 else 190.0
        marking((x, 45.0), (x, tip), interruption if index == 1 else None)
    marking((35.0, 45.0), (310.0, 45.0), None)
    if rear:
        marking((35.0, 190.0), (310.0, 190.0), None)
    if continuation:
        marking((125.0, 200.0), (125.0, 228.0), None)
    data = tuple(pairs)
    topology = find_topology_candidates(data)
    support = measure_topology_support(topology, data, groups)
    return assemble_slots(CameraId("camera"), 380, 240, data, topology, support, groups)


def test_shared_rails_form_three_convex_projected_slots() -> None:
    candidates = [item for item in _row() if item.proposal]
    assert len(candidates) == 3
    assert all(
        item.status == "observed" and len(item.junction_ids) == 4 for item in candidates
    )
    assert all(
        all(boundary.kind == "observed" for boundary in item.boundaries)
        for item in candidates
    )
    corners = {
        point
        for item in candidates
        if item.proposal
        for point in item.proposal.polygon.vertices
    }
    assert Point(65 / 379, 45 / 239) in corners
    assert Point(275 / 379, 190 / 239) in corners


def test_aligned_tips_estimate_only_missing_rear_boundary() -> None:
    candidates = [item for item in _row(rear=False) if item.proposal]
    assert len(candidates) == 3
    for item in candidates:
        assert item.status == "estimated"
        (back,) = [
            boundary for boundary in item.boundaries if boundary.kind == "estimated"
        ]
        assert back.start[1] == pytest.approx(190.0)
        assert back.end[1] == pytest.approx(190.0)
        assert len(back.line_ids) == 3


def test_short_occlusion_is_retained_as_estimated_source_gap() -> None:
    candidates = [item for item in _row(interruption=(108, 125)) if item.proposal]
    assert len(candidates) == 3
    estimated = [item for item in candidates if item.status == "estimated"]
    assert len(estimated) == 2
    for item in estimated:
        (gap,) = [
            boundary for boundary in item.boundaries if boundary.kind == "estimated"
        ]
        assert gap.line_ids == (1,)
        assert dist(gap.start, gap.end) == pytest.approx(17.0)


def test_bracketed_short_tip_extension_is_bounded() -> None:
    candidates = [item for item in _row(rear=False, shortened_tip=178) if item.proposal]
    assert len(candidates) == 3
    extensions = [
        dist(boundary.start, boundary.end)
        for item in candidates
        for boundary in item.boundaries
        if boundary.kind == "estimated" and boundary.line_ids == (1,)
    ]
    assert extensions == pytest.approx([12, 12])


@pytest.mark.parametrize("length", [35, 55])
def test_long_unsupported_gap_does_not_create_physical_roi(length: int) -> None:
    candidates = _row(interruption=(90, 90 + length))
    assert sum(item.proposal is not None for item in candidates) == 1
    assert any(
        item.status == "unresolved" and "Insufficient source" in item.reason
        for item in candidates
    )


def test_truncated_separator_does_not_merge_neighboring_spaces() -> None:
    candidates = _row(rear=False, shortened_tip=83)
    assert sum(item.proposal is not None for item in candidates) == 1
    assert any(item.status == "unresolved" for item in candidates)


def test_two_separators_cannot_corroborate_a_row() -> None:
    assert all(item.proposal is None for item in _row(separator_count=2))


def test_apparent_tip_with_observed_continuation_is_not_a_rear_corner() -> None:
    candidates = _row(rear=False, continuation=True)
    assert sum(item.proposal is not None for item in candidates) < 3
    assert any("continuation" in item.reason for item in candidates)

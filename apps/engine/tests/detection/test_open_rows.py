"""Source-backed open row structure without inventing physical slot depth."""

from dataclasses import dataclass, field
from math import hypot

import pytest

from parkvision.detection.initialization import RowEvidence, SlotCandidate
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_topology import find_topology_candidates
from parkvision.detection.row_reconstruction import reconstruct_open_rows
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import PixelPoint, point_line_distance
from parkvision.detection.slot_assembly import assemble_slots
from parkvision.detection.topology_candidate import TopologyCandidates
from parkvision.detection.topology_support import IntersectionSourceSupport
from parkvision.detection.topology_support_measurement import measure_topology_support
from parkvision.domain.types import CameraId


@dataclass
class _Scene:
    pairs: list[MarkingPair] = field(default_factory=list)
    context: list[LineSegment] = field(default_factory=list)
    sources: dict[int, SegmentGroup] = field(default_factory=dict)

    def paint(self, start: PixelPoint, end: PixelPoint, width: float = 4.0) -> int:
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
        self.pairs.append(MarkingPair((edges[0], edges[1]), axis, (width, width), 1.0))
        return axis_id

    def boundary(self, start: PixelPoint, end: PixelPoint) -> int:
        line_id = 5000 + len(self.context)
        line = LineSegment(line_id, start, end)
        self.context.append(line)
        self.sources[line_id] = SegmentGroup(
            (LineSegment(10000 + line_id, start, end),)
        )
        return line_id

    def topology(
        self,
    ) -> tuple[TopologyCandidates, tuple[IntersectionSourceSupport, ...]]:
        pairs = tuple(self.pairs)
        topology = find_topology_candidates(
            pairs, tuple(self.context), min_context_length_px=20.0
        )
        return topology, measure_topology_support(topology, pairs, self.sources)

    def reconstruct(
        self,
        existing_rows: tuple[RowEvidence, ...] = (),
        existing_candidates: tuple[SlotCandidate, ...] = (),
    ) -> tuple[tuple[RowEvidence, ...], tuple[SlotCandidate, ...]]:
        topology, support = self.topology()
        return reconstruct_open_rows(
            440,
            340,
            tuple(self.pairs),
            topology,
            support,
            self.sources,
            existing_rows=existing_rows,
            existing_candidates=existing_candidates,
        )


def _discrete_t_row(
    tips: tuple[float, ...] = (205.0, 220.0, 245.0, 265.0),
) -> tuple[_Scene, tuple[int, ...]]:
    scene = _Scene()
    stems = []
    for x, y, tip in zip(
        (60.0, 130.0, 215.0, 300.0), (50.0, 60.0, 74.0, 86.0), tips, strict=True
    ):
        stems.append(scene.paint((x, y), (x + 5.0, tip)))
        scene.paint((x - 16.0, y), (x + 16.0, y))
    return scene, tuple(stems)


def _assert_partial_only(candidates: tuple[SlotCandidate, ...]) -> None:
    assert candidates
    assert all(candidate.status == "unresolved" for candidate in candidates)
    assert all(candidate.proposal is None for candidate in candidates)
    assert all(candidate.geometry_kind == "partial" for candidate in candidates)
    assert all(candidate.rejection_code is None for candidate in candidates)
    assert all(candidate.missing_boundaries for candidate in candidates)
    assert all(candidate.boundaries for candidate in candidates)


def test_distinct_t_caps_form_a_local_row_without_a_shared_transverse_line() -> None:
    scene, stems = _discrete_t_row()

    rows, partial = scene.reconstruct()

    matching = [row for row in rows if set(stems) <= set(row.separator_ids)]
    assert matching
    assert any(row.geometry_kind == "open_ended" for row in matching)
    assert len(partial) >= 3
    _assert_partial_only(partial)
    assert all(candidate.separator_ids for candidate in partial)


def test_stepped_boundary_segments_anchor_one_partial_row_without_curb_semantics() -> (
    None
):
    scene = _Scene()
    stems = []
    boundary_ids = []
    for y, rear_x in ((55.0, 140.0), (115.0, 155.0), (185.0, 172.0), (260.0, 192.0)):
        stems.append(scene.paint((70.0 + 0.12 * y, y), (rear_x, y)))
        boundary_ids.append(
            scene.boundary((rear_x + 2.0, y - 20.0), (rear_x - 2.0, y + 20.0))
        )

    rows, partial = scene.reconstruct()

    matching = [row for row in rows if set(stems) <= set(row.separator_ids)]
    assert matching
    assert any(
        set(boundary_ids) <= {line_id for ids in row.anchor_line_ids for line_id in ids}
        for row in matching
    )
    assert all("curb" not in kind for row in matching for kind in row.anchor_kinds)
    _assert_partial_only(partial)


def test_unequal_anchor_spacing_is_retained_in_the_row_evidence() -> None:
    scene, stems = _discrete_t_row()

    rows, partial = scene.reconstruct()

    row = next(row for row in rows if set(stems) <= set(row.separator_ids))
    positions = {
        line_id: point for line_id, point in zip(row.separator_ids, row.positions)
    }
    observed_x = [positions[line_id][0] for line_id in stems]
    assert observed_x == pytest.approx((60.0, 130.0, 215.0, 300.0), abs=2.0)
    _assert_partial_only(partial)


def test_truncated_separator_ends_remain_partial_and_source_bounded() -> None:
    scene, _ = _discrete_t_row((205.0, 170.0, 185.0, 265.0))

    _, partial = scene.reconstruct()

    _assert_partial_only(partial)
    axes = {pair.centerline.segment_id: pair.centerline for pair in scene.pairs}
    for candidate in partial:
        for boundary in candidate.boundaries:
            assert boundary.kind == "observed"
            for line_id in boundary.line_ids:
                if line_id not in axes:
                    continue
                axis = axes[line_id]
                unit = (
                    (axis.end[0] - axis.start[0]) / axis.length,
                    (axis.end[1] - axis.start[1]) / axis.length,
                )
                for point in (boundary.start, boundary.end):
                    assert point_line_distance(point, axis) < 1e-6
                    along = (point[0] - axis.start[0]) * unit[0] + (
                        point[1] - axis.start[1]
                    ) * unit[1]
                    assert -1e-6 <= along <= axis.length + 1e-6


@pytest.mark.parametrize("kind", ["curb-only", "two-strokes", "zebra"])
def test_unrelated_boundaries_and_stripes_do_not_become_partial_parking_bays(
    kind: str,
) -> None:
    scene = _Scene()
    if kind == "curb-only":
        scene.boundary((40.0, 50.0), (360.0, 90.0))
    elif kind == "two-strokes":
        for x in (80.0, 190.0):
            scene.paint((x, 60.0), (x, 220.0))
    else:
        for y in (60.0, 88.0, 116.0, 144.0, 172.0):
            scene.paint((70.0, y), (240.0, y), width=12.0)

    _, partial = scene.reconstruct()

    assert partial == ()


def test_one_reused_context_line_is_not_independent_anchor_corroboration() -> None:
    scene = _Scene()
    for y in (60.0, 120.0, 180.0, 240.0):
        scene.paint((70.0, y), (190.0, y))
    # Four endpoint intersections with one physical edge remain one source of
    # evidence. This is also a plausible curb or vehicle-side pattern.
    scene.boundary((70.0, 30.0), (70.0, 270.0))

    _, partial = scene.reconstruct()

    assert partial == ()


def test_isolated_anchor_does_not_promote_an_otherwise_aligned_chain() -> None:
    scene = _Scene()
    for x, y in ((60.0, 50.0), (130.0, 60.0), (215.0, 74.0), (300.0, 86.0)):
        scene.paint((x, y), (x + 5.0, y + 150.0))
    # One unrelated T junction can occur next to repeated object edges. The
    # other separator hypotheses have no independent local anchor evidence.
    scene.paint((44.0, 50.0), (76.0, 50.0))

    _, partial = scene.reconstruct()

    assert partial == ()


def test_collinear_paint_fragments_across_islands_are_not_adjacent_separators() -> None:
    scene = _Scene()
    for x in (40.0, 155.0, 270.0):
        scene.paint((x, 110.0), (x + 75.0, 110.0))
        scene.paint((x, 94.0), (x, 126.0))
    _, partial = scene.reconstruct()
    assert partial == ()


def test_partial_source_pair_is_not_duplicated_by_alternate_anchor_rejections() -> None:
    scene, stems = _discrete_t_row()
    existing = tuple(
        SlotCandidate(
            "unresolved",
            None,
            (),
            (node,),
            "Separator tip is clipped",
            separator_ids=stems[:2],
        )
        for node in (200, 201)
    )
    _, partial = scene.reconstruct(existing_candidates=existing)
    pairs = [frozenset(candidate.separator_ids) for candidate in partial]
    assert len(pairs) == len(set(pairs)) == 3
    merged = next(
        c for c in partial if frozenset(c.separator_ids) == frozenset(stems[:2])
    )
    assert {200, 201} <= set(merged.junction_ids)


def test_abrupt_direction_change_ends_a_local_row() -> None:
    scene = _Scene()
    first_direction = set()
    second_direction = set()
    for x in (60.0, 130.0):
        first_direction.add(scene.paint((x, 55.0), (x, 180.0)))
        scene.paint((x - 16.0, 55.0), (x + 16.0, 55.0))
    for y in (230.0, 290.0):
        second_direction.add(scene.paint((230.0, y), (360.0, y)))
        scene.paint((230.0, y - 16.0), (230.0, y + 16.0))

    rows, partial = scene.reconstruct()

    assert all(
        not (
            set(row.separator_ids) & first_direction
            and set(row.separator_ids) & second_direction
        )
        for row in rows
    )
    assert all(candidate.proposal is None for candidate in partial)


def test_closed_row_keeps_existing_full_polygons_and_does_not_duplicate_them() -> None:
    scene = _Scene()
    for x in (65.0, 135.0, 220.0, 305.0):
        scene.paint((x, 55.0), (x, 240.0))
    scene.paint((35.0, 55.0), (340.0, 55.0))
    scene.paint((35.0, 240.0), (340.0, 240.0))
    topology, support = scene.topology()
    existing_rows = []
    full = assemble_slots(
        CameraId("camera"),
        440,
        340,
        tuple(scene.pairs),
        topology,
        support,
        scene.sources,
        row_diagnostics=existing_rows,
    )
    assert sum(candidate.proposal is not None for candidate in full) == 3

    _, partial = scene.reconstruct(tuple(existing_rows), full)

    occupied = {
        frozenset(candidate.separator_ids) for candidate in full if candidate.proposal
    }
    assert all(
        frozenset(candidate.separator_ids) not in occupied for candidate in partial
    )


def test_hatched_pattern_with_two_perimeters_never_produces_a_full_roi() -> None:
    scene = _Scene()
    for y in (75.0, 115.0, 155.0):
        scene.paint((55.0, y), (140.0, y + 50.0))
    scene.paint((55.0, 45.0), (55.0, 260.0))
    scene.paint((155.0, 45.0), (155.0, 280.0))
    topology, support = scene.topology()
    existing_rows = []
    full = assemble_slots(
        CameraId("camera"),
        440,
        340,
        tuple(scene.pairs),
        topology,
        support,
        scene.sources,
        row_diagnostics=existing_rows,
    )

    _, partial = scene.reconstruct(tuple(existing_rows), full)

    assert all(candidate.proposal is None for candidate in (*full, *partial))

from dataclasses import replace
from itertools import permutations, product
from math import hypot

import pytest

from parkvision.detection import marking_topology
from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_pairing import pair_marking_edges
from parkvision.detection.marking_topology import find_topology_candidates
from parkvision.detection.segment_group import SegmentGroup


def axis(index: int, start: tuple, end: tuple) -> MarkingPair:
    ux, uy = end[0] - start[0], end[1] - start[1]
    length = hypot(ux, uy)
    nx, ny = -3 * uy / length, 3 * ux / length
    first_edge, second_edge = (
        LineSegment(
            2 * index + i,
            (start[0] + sign * nx, start[1] + sign * ny),
            (end[0] + sign * nx, end[1] + sign * ny),
        )
        for i, sign in enumerate((-1, 1))
    )
    return MarkingPair(
        (first_edge, second_edge), LineSegment(10000 + index, start, end), (6, 6), 1
    )


@pytest.mark.parametrize("unit", [(1, 0), (0, 1), (0.6, 0.8), (-0.8, 0.6)])
@pytest.mark.parametrize(
    ("kind", "first_start", "second_start", "roles"),
    [
        ("L", 0, 0, ("end", "end")),
        ("T", -60, 0, ("interior", "end")),
        ("X", -60, -60, ("interior", "interior")),
    ],
)
def test_topology_under_rotation_endpoint_reversal_and_input_permutation(
    unit: tuple, kind: str, first_start: int, second_start: int, roles: tuple
) -> None:
    ux, uy = unit

    def point(x: float, y: float) -> tuple[float, float]:
        return 200 + x * ux - y * uy, 200 + x * uy + y * ux

    for reverse_first, reverse_second in product((False, True), repeat=2):
        first = (point(first_start, 0), point(60, 0))
        second = (point(0, second_start), point(0, 60))
        pairs = (
            axis(0, *first[:: -1 if reverse_first else 1]),
            axis(1, *second[:: -1 if reverse_second else 1]),
        )
        for order in permutations(pairs):
            result = find_topology_candidates(order)
            (node,) = result.junctions
            (hit,) = node.intersections
            assert node.position == pytest.approx((200, 200))
            assert hit.position == pytest.approx((200, 200))
            assert hit.topology == kind
            assert hit.roles == roles
            assert hit.line_ids == (10000, 10001)
            assert hit.extension_px == (0, 0)
            assert not hit.hypothetical and not hit.uses_unpaired_context
            assert len(result.endpoints) == 4
            assert sum(bool(e.junction_ids) for e in result.endpoints) == roles.count(
                "end"
            )


def test_endpoint_tolerance_changes_arm_count_but_never_extends_geometry() -> None:
    first = axis(0, (-60, 0), (2, 0))
    second = axis(1, (0, -60), (0, 60))
    assert (
        find_topology_candidates((first, second), endpoint_tolerance_px=2)
        .junctions[0]
        .intersections[0]
        .topology
        == "T"
    )
    assert (
        find_topology_candidates((first, second), endpoint_tolerance_px=1.99)
        .junctions[0]
        .intersections[0]
        .topology
        == "X"
    )
    outside = axis(2, (5, -60), (5, 60))
    assert not find_topology_candidates(
        (first, outside), endpoint_tolerance_px=100
    ).junctions


def test_explicit_extension_limit_applies_to_both_lines_and_marks_even_small_gaps() -> (
    None
):
    pairs = (axis(0, (0, 0), (60, 0)), axis(1, (64, 3), (64, 60)))
    assert not find_topology_candidates(pairs).junctions
    assert not find_topology_candidates(pairs, max_extension_px=3.999).junctions
    (node,) = find_topology_candidates(pairs, max_extension_px=4).junctions
    (hit,) = node.intersections
    assert hit.position == pytest.approx((64, 0))
    assert hit.extension_px == pytest.approx((4, 3))
    assert hit.topology == "L" and hit.hypothetical
    near = axis(2, (60.1, -60), (60.1, 60))
    hit = (
        find_topology_candidates((pairs[0], near), max_extension_px=0.2)
        .junctions[0]
        .intersections[0]
    )
    assert hit.topology == "T" and hit.hypothetical
    assert hit.extension_px == pytest.approx((0.1, 0))


@pytest.mark.parametrize("second", [((0, 4), (60, 4)), ((10, 0), (70, 0))])
def test_parallel_and_collinear_segments_have_no_unique_node(second: tuple) -> None:
    assert not find_topology_candidates(
        (axis(0, (0, 0), (60, 0)), axis(1, *second)), max_extension_px=50
    ).junctions


def test_nearly_parallel_is_rejected_before_remote_unstable_intersection() -> None:
    pairs = (axis(0, (-60, 0), (60, 0)), axis(1, (-60, -0.01), (60, 0.01)))
    assert not find_topology_candidates(pairs).junctions
    assert len(find_topology_candidates(pairs, min_crossing_angle=0.001).junctions) == 1


def test_isolated_off_frame_and_short_lines_keep_objects_ids_and_all_endpoints() -> (
    None
):
    pair = axis(0, (-50, -20), (-1, -20))
    context = LineSegment(99, (10, 10), (11, 10))
    result = find_topology_candidates((pair,), (context,))
    assert result.lines[0] is context
    assert result.lines[1] is pair.centerline
    assert result.unpaired_line_ids == (99,)
    assert result.junctions == ()
    assert [e.position for e in result.endpoints] == [
        context.start,
        context.end,
        pair.centerline.start,
        pair.centerline.end,
    ]
    assert all(e.junction_ids == () for e in result.endpoints)
    assert find_topology_candidates(()).lines == ()


def test_context_is_only_used_at_axis_ends_never_against_other_context() -> None:
    pair = axis(0, (0, 0), (60, 0))
    curb = LineSegment(99, (60, -30), (60, 30))
    interior = LineSegment(100, (30, -30), (30, 30))
    other_context = LineSegment(101, (40, -20), (80, 20))
    result = find_topology_candidates((pair,), (curb, interior, other_context))
    (node,) = result.junctions
    assert len(node.intersections) == 2
    assert all(hit.uses_unpaired_context for hit in node.intersections)
    assert all(hit.topology == "T" for hit in node.intersections)
    assert {hit.line_ids for hit in node.intersections} == {(99, 10000), (101, 10000)}
    assert not find_topology_candidates((), (curb, interior, other_context)).junctions
    assert len(result.lines) == 4 and len(result.endpoints) == 8
    assert next(
        e for e in result.endpoints if e.line_id == 10000 and e.end == "end"
    ).junction_ids == (0,)


def test_short_context_only_changes_participation_not_retention() -> None:
    pair = axis(0, (0, 0), (60, 0))
    context = LineSegment(99, (60, 0), (60, 20))
    normal = find_topology_candidates((pair,), (context,))
    permissive = find_topology_candidates((pair,), (context,), min_context_length_px=20)
    assert normal.lines == permissive.lines and len(normal.endpoints) == 4
    assert not normal.junctions
    assert permissive.junctions[0].intersections[0].topology == "L"


def test_lines_too_short_to_resolve_an_arm_keep_endpoints_without_false_labels() -> (
    None
):
    pairs = (axis(0, (-1, 0), (1, 0)), axis(1, (0, -60), (0, 60)))
    result = find_topology_candidates(pairs)
    assert not result.junctions and len(result.endpoints) == 4
    (node,) = find_topology_candidates(pairs, endpoint_tolerance_px=0).junctions
    assert node.intersections[0].topology == "X"


def test_context_ids_roles_and_extension_stay_aligned_when_context_id_comes_first() -> (
    None
):
    pair = axis(0, (0, 0), (60, 0))
    context = LineSegment(99, (63, -30), (63, 30))
    (node,) = find_topology_candidates(
        (pair,), (context,), max_extension_px=3
    ).junctions
    (hit,) = node.intersections
    assert hit.line_ids == (99, 10000)
    assert hit.roles == ("interior", "end")
    assert hit.extension_px == (0, 3)
    assert hit.hypothetical and hit.uses_unpaired_context


def test_deduplication_preserves_all_pairwise_hits_and_stable_order() -> None:
    pairs = (
        axis(0, (-60, 0), (60, 0)),
        axis(1, (0, -60), (0, 60)),
        axis(2, (-60, -60), (60, 60)),
    )
    expected = find_topology_candidates(pairs)
    assert len(expected.junctions) == 1
    assert len(expected.junctions[0].intersections) == 3
    for order in permutations(pairs):
        assert find_topology_candidates(order) == expected


def test_deduplication_does_not_chain_or_promote_hypothetical_hits() -> None:
    pairs = (
        axis(0, (-60, 0), (60, 0)),
        axis(1, (0, -60), (0, 60)),
        axis(2, (2, -60), (2, 60)),
        axis(3, (4, -60), (4, 60)),
    )
    result = find_topology_candidates(pairs, dedup_radius_px=3)
    assert [node.position for node in result.junctions] == [(0, 0), (4, 0)]
    assert [len(node.intersections) for node in result.junctions] == [2, 1]
    assert len(find_topology_candidates(pairs, dedup_radius_px=0).junctions) == 3
    # A speculative hit at x=0 is close to a real-extent hit at x=2.
    hypothetical_pairs = (pairs[0], axis(1, (0, 1), (0, 60)), pairs[2])
    (node,) = find_topology_candidates(hypothetical_pairs, max_extension_px=1).junctions
    assert node.position == (2, 0)
    assert [hit.hypothetical for hit in node.intersections] == [False, True]


def test_gap_in_source_coverage_remains_a_geometric_candidate_not_confirmation() -> (
    None
):
    sources = (
        LineSegment(0, (-60, -3), (-4, -3)),
        LineSegment(1, (4, -3), (60, -3)),
        LineSegment(2, (-60, 3), (-4, 3)),
        LineSegment(3, (4, 3), (60, 3)),
        LineSegment(4, (-3, -60), (-3, 60)),
        LineSegment(5, (3, -60), (3, 60)),
    )
    groups = tuple(
        SegmentGroup(s) for s in (sources[:2], sources[2:4], sources[4:5], sources[5:])
    )
    fitted = tuple(fit_group_line(g, segment_id=6 + i) for i, g in enumerate(groups))
    pairs = pair_marking_edges(fitted, min_width_px=2, max_width_px=12)
    before = sources, groups, fitted, pairs
    result = find_topology_candidates(pairs)
    (hit,) = result.junctions[0].intersections
    assert hit.topology == "X" and not hit.hypothetical
    assert hit.position == (0, 0)  # Both horizontal source edges have a gap here.
    assert (sources, groups, fitted, pairs) == before
    assert all(any(line is p.centerline for p in pairs) for line in result.lines)


@pytest.mark.parametrize(
    "parameter",
    [
        "endpoint_tolerance_px",
        "max_extension_px",
        "dedup_radius_px",
        "min_context_length_px",
    ],
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_validates_all_tolerances_even_without_lines(
    parameter: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        find_topology_candidates((), (), **{parameter: value})


@pytest.mark.parametrize("value", [-1, 0, 90.1, float("nan"), float("inf")])
def test_validates_crossing_angle(value: float) -> None:
    with pytest.raises(ValueError, match="min_crossing_angle"):
        find_topology_candidates((), min_crossing_angle=value)


@pytest.mark.parametrize(
    "start,end",
    [
        ((1, 1), (1, 1)),
        ((float("nan"), 0), (1, 1)),
        ((0, 0), (float("inf"), 1)),
        ((-1e308, 0), (1e308, 0)),
    ],
)
def test_rejects_invalid_geometry_even_in_unused_context(
    start: tuple, end: tuple
) -> None:
    with pytest.raises(ValueError, match="finite, non-zero"):
        find_topology_candidates((), (LineSegment(0, start, end),))


def test_rejects_id_collisions_and_invalid_paired_sources() -> None:
    pair = axis(0, (0, 0), (60, 0))
    with pytest.raises(ValueError, match="unique"):
        find_topology_candidates((pair, pair))
    with pytest.raises(ValueError, match="unique"):
        find_topology_candidates((pair,), (pair.edges[0],))
    with pytest.raises(ValueError, match="unique"):
        find_topology_candidates((pair,), (pair.centerline,))
    bad_edge = replace(pair.edges[0], end=pair.edges[0].start)
    with pytest.raises(ValueError, match="finite, non-zero"):
        find_topology_candidates((replace(pair, edges=(bad_edge, pair.edges[1])),))


def test_rejects_overflow_of_extended_finite_geometry() -> None:
    line = LineSegment(99, (1e308, 0), (1.2e308, 0))
    with pytest.raises(ValueError, match="Extended geometry"):
        find_topology_candidates((), (line,), max_extension_px=1e308)


def test_perpendicular_angle_boundary_is_inclusive() -> None:
    pairs = (axis(0, (-60, 0), (60, 0)), axis(1, (0, -60), (0, 60)))
    assert (
        find_topology_candidates(pairs, min_crossing_angle=90)
        .junctions[0]
        .intersections[0]
        .topology
        == "X"
    )


def test_sparse_thousands_of_lines_skip_exact_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object) -> float:
        pytest.fail("Distant bboxes should not reach angle/intersection arithmetic")

    monkeypatch.setattr(marking_topology, "angle_difference", unexpected)
    pairs = tuple(axis(i, (i * 200, 0), (i * 200 + 60, 0)) for i in range(2000))
    result = find_topology_candidates(pairs)
    assert not result.junctions
    assert len(result.lines) == 2000 and len(result.endpoints) == 4000

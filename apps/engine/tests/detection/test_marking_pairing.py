from itertools import permutations, product

import pytest

from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_pairing import pair_marking_edges
from parkvision.detection.segment_merging import group_segments, projected_segment_gap


def pair(*lines: LineSegment, **parameters: float) -> tuple[MarkingPair, ...]:
    settings = {"min_width_px": 2.0, "max_width_px": 12.0}
    settings.update(parameters)
    return pair_marking_edges(lines, **settings)


def test_legacy_model_import_is_the_shared_result_type() -> None:
    from parkvision.detection.marking_pairing import MarkingPair as LegacyMarkingPair

    assert LegacyMarkingPair is MarkingPair
    (result,) = pair(
        LineSegment(0, (0.0, 0.0), (100.0, 0.0)),
        LineSegment(1, (0.0, 6.0), (100.0, 6.0)),
    )
    assert type(result) is MarkingPair


def test_parallel_edges_use_only_their_common_projected_extent() -> None:
    first = LineSegment(10, (0.0, 3.0), (100.0, 3.0))
    second = LineSegment(20, (20.0, 11.0), (120.0, 11.0))

    (result,) = pair(first, second)

    assert result.edges[0] is first
    assert result.edges[1] is second
    assert result.centerline.start == pytest.approx((20.0, 7.0))
    assert result.centerline.end == pytest.approx((100.0, 7.0))
    assert result.widths_px == pytest.approx((8.0, 8.0))
    assert result.overlap_ratio == pytest.approx(0.8)
    assert result.centerline.segment_id > 20


@pytest.mark.parametrize("unit", [(1, 0), (0, 1), (0.6, 0.8), (0.8, -0.6), (-0.6, 0.8)])
def test_pairing_is_axial_and_invariant_to_endpoint_and_input_order(
    unit: tuple[float, float],
) -> None:
    ux, uy = unit

    def point(t: float, q: float) -> tuple[float, float]:
        return 300 + t * ux - q * uy, 250 + t * uy + q * ux

    for order in permutations((10, 20)):
        for reverse_first, reverse_second in product((False, True), repeat=2):
            first_points = (point(0, 0), point(100, 0))
            second_points = (point(20, 8), point(120, 8))
            first = LineSegment(10, *first_points[:: -1 if reverse_first else 1])
            second = LineSegment(20, *second_points[:: -1 if reverse_second else 1])
            lines = {10: first, 20: second}

            (result,) = pair(*(lines[i] for i in order))

            assert result.centerline.start == pytest.approx(point(20, 4))
            assert result.centerline.end == pytest.approx(point(100, 4))
            assert result.widths_px == pytest.approx((8, 8))
            assert result.overlap_ratio == pytest.approx(0.8)
            assert result.edges[0] is first and result.edges[1] is second


def test_rejects_incompatible_angles_even_with_permissive_width_bounds() -> None:
    first = LineSegment(0, (0.0, 0.0), (100.0, 0.0))
    second = LineSegment(1, (0.0, 5.0), (100.0, 12.0))

    assert pair(first, second, max_width_px=20, max_width_ratio=3) == ()
    assert len(pair(first, second, angle_tolerance=5, max_width_ratio=3)) == 1


@pytest.mark.parametrize("start", [100.0, 101.0, 200.0])
def test_rejects_nonoverlapping_or_touching_extents(start: float) -> None:
    assert (
        pair(
            LineSegment(0, (0.0, 0.0), (100.0, 0.0)),
            LineSegment(1, (start, 6.0), (start + 100.0, 6.0)),
        )
        == ()
    )


def test_requires_absolute_and_relative_overlap_of_the_longer_edge() -> None:
    first = LineSegment(0, (0.0, 0.0), (100.0, 0.0))
    short = LineSegment(1, (10.0, 6.0), (30.0, 6.0))
    partial = LineSegment(2, (60.0, 6.0), (160.0, 6.0))
    assert pair(first, short, min_overlap_ratio=0.1) == ()
    assert pair(first, partial) == ()
    assert len(pair(first, partial, min_overlap_ratio=0.4)) == 1
    assert pair(first, partial, min_overlap_ratio=0.4, min_overlap_px=41) == ()
    # A shorter edge entirely inside a long one has ratio 0.4, not 1.0.
    contained = LineSegment(3, (20.0, 6.0), (60.0, 6.0))
    assert pair(first, contained) == ()


@pytest.mark.parametrize(
    ("width", "accepted"),
    [
        (-8.0, True),
        (0.0, False),
        (1.999, False),
        (2.0, True),
        (12.0, True),
        (12.001, False),
        (80.0, False),
    ],
)
def test_explicit_pixel_width_bounds(width: float, accepted: bool) -> None:
    result = pair(
        LineSegment(0, (0.0, 0.0), (100.0, 0.0)),
        LineSegment(1, (0.0, width), (100.0, width)),
    )
    assert bool(result) is accepted


def test_mild_perspective_uses_cross_section_midpoints() -> None:
    first = LineSegment(0, (0.0, -3.0), (100.0, -4.0))
    second = LineSegment(1, (0.0, 3.0), (100.0, 4.0))

    (result,) = pair(first, second)

    assert result.centerline.start == pytest.approx((0, 0))
    assert result.centerline.end == pytest.approx((100, 0))
    assert result.widths_px == pytest.approx((6, 8))
    assert pair(first, second, max_width_ratio=1.2) == ()


@pytest.mark.parametrize(("narrow", "wide"), [(1.0, 9.0), (6.0, 14.0)])
def test_checks_width_at_both_shared_ends_not_just_the_midpoint(
    narrow: float,
    wide: float,
) -> None:
    first = LineSegment(0, (0.0, -narrow / 2), (1000.0, -wide / 2))
    second = LineSegment(1, (0.0, narrow / 2), (1000.0, wide / 2))
    assert pair(first, second, max_width_ratio=10) == ()


def test_checks_width_only_within_overlap_without_extrapolation() -> None:
    # The longer edge would be too close at x=0 and too far at x=2000.
    first = LineSegment(0, (0.0, 0.0), (2000.0, 0.0))
    second = LineSegment(1, (100.0, 3.0), (200.0, 4.0))

    (result,) = pair(first, second, min_overlap_ratio=0.04)

    assert min(result.widths_px) >= 2
    assert max(result.widths_px) <= 12
    assert 99 < result.centerline.start[0] < result.centerline.end[0] < 201


def test_rejects_crossing_edges_even_when_endpoint_widths_are_valid() -> None:
    assert (
        pair(
            LineSegment(0, (0.0, 0.0), (1000.0, 0.0)),
            LineSegment(1, (0.0, 6.0), (1000.0, -6.0)),
        )
        == ()
    )


def test_conflicts_use_overlap_then_width_and_each_edge_only_once() -> None:
    first = LineSegment(5, (0.0, 0.0), (100.0, 0.0))
    short_close = LineSegment(8, (0.0, 3.0), (70.0, 3.0))
    full = LineSegment(12, (0.0, 8.0), (100.0, 8.0))
    other = LineSegment(20, (0.0, 30.0), (100.0, 30.0))
    other_partner = LineSegment(30, (0.0, 36.0), (100.0, 36.0))
    inputs = (first, short_close, full, other, other_partner)

    expected = pair(*inputs)

    assert {tuple(e.segment_id for e in p.edges) for p in expected} == {
        (5, 12),
        (20, 30),
    }
    used = [e.segment_id for p in expected for e in p.edges]
    assert len(used) == len(set(used))
    assert len({p.centerline.segment_id for p in expected}) == len(expected)
    assert all(p.centerline.segment_id > 30 for p in expected)
    for order in permutations(inputs):
        assert pair(*order) == expected


def test_equal_overlap_prefers_nearer_partner_and_id_breaks_exact_ties() -> None:
    first = LineSegment(1, (0.0, 0.0), (100.0, 0.0))
    near = LineSegment(2, (0.0, 4.0), (100.0, 4.0))
    far = LineSegment(3, (0.0, 10.0), (100.0, 10.0))
    assert pair(first, near, far)[0].edges == (first, near)
    tied = LineSegment(4, (0.0, -4.0), (100.0, -4.0))
    assert pair(first, tied, near)[0].edges == (first, near)


def test_empty_and_single_edges_are_valid_unpaired_inputs() -> None:
    assert pair() == ()
    assert pair(LineSegment(99, (0, 0), (100, 0))) == ()


def test_preserves_source_groups_fragments_gaps_and_unpaired_edge() -> None:
    sources = (
        LineSegment(0, (0, 0), (40, 0)),
        LineSegment(1, (48, 0), (100, 0)),
        LineSegment(2, (0, 8), (40, 8)),
        LineSegment(3, (48, 8), (100, 8)),
        LineSegment(4, (0, 50), (100, 50)),
    )
    groups = group_segments(sources, 3, 2, 12)
    fitted = tuple(fit_group_line(g, segment_id=5 + i) for i, g in enumerate(groups))
    before = (sources, groups, fitted)

    (result,) = pair(*fitted)

    assert result.edges[0] is fitted[0] and result.edges[1] is fitted[1]
    assert result.centerline.start == pytest.approx((0, 4))
    assert result.centerline.end == pytest.approx((100, 4))
    assert (sources, groups, fitted) == before
    assert projected_segment_gap(*groups[0].segments) == 8
    assert projected_segment_gap(*groups[1].segments) == 8
    assert groups[2].segments[0] is sources[4]  # The single edge is retained.
    assert all(
        any(s is member for g in groups for member in g.segments) for s in sources
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_width_px": 0},
        {"min_width_px": -1},
        {"max_width_px": 1},
        {"max_width_px": float("inf")},
        {"min_width_px": float("nan")},
        {"angle_tolerance": -1},
        {"angle_tolerance": 90},
        {"angle_tolerance": float("nan")},
        {"min_overlap_px": 0},
        {"min_overlap_px": float("inf")},
        {"min_overlap_ratio": 0},
        {"min_overlap_ratio": 1.1},
        {"min_overlap_ratio": float("nan")},
        {"max_width_ratio": 0.9},
        {"max_width_ratio": float("inf")},
    ],
)
def test_validates_parameters_even_without_edges(parameters: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        pair(**parameters)


@pytest.mark.parametrize(
    "line",
    [
        LineSegment(0, (1, 1), (1, 1)),
        LineSegment(0, (float("nan"), 1), (2, 2)),
        LineSegment(0, (0, 0), (float("inf"), 2)),
        LineSegment(0, (-1e308, 0), (1e308, 0)),
    ],
)
def test_rejects_invalid_geometry_even_for_a_single_edge(line: LineSegment) -> None:
    with pytest.raises(ValueError, match="finite, non-zero"):
        pair(line)


def test_rejects_duplicate_edge_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        pair(LineSegment(0, (0, 0), (100, 0)), LineSegment(0, (0, 6), (100, 6)))


def test_axis_ids_are_above_even_short_unpaired_edges() -> None:
    (result,) = pair(
        LineSegment(0, (0, 0), (100, 0)),
        LineSegment(1, (0, 6), (100, 6)),
        LineSegment(100, (0, 20), (1, 20)),
    )
    assert result.centerline.segment_id > 100

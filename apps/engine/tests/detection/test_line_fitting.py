from itertools import permutations, product
from math import isfinite

import pytest

from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import (
    group_segments,
    point_line_distance,
    projected_segment_gap,
)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param((1.25, 3.5), (11.25, 3.5), id="horizontal"),
        pytest.param((2.0, 1.0), (2.0, 11.0), id="vertical"),
        pytest.param((-2.0, -4.0), (8.0, 6.0), id="diagonal"),
        pytest.param((0.0, 12.0), (10.0, 2.0), id="negative-slope"),
        pytest.param((8.0, 0.0), (2.0, 30.0), id="steep-negative-slope"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_singleton_preserves_geometry_with_new_identity(
    start: tuple[float, float], end: tuple[float, float], reverse: bool
) -> None:
    segment = LineSegment(7, end if reverse else start, start if reverse else end)

    fitted = fit_group_line(SegmentGroup((segment,)), segment_id=8)

    assert fitted.start == pytest.approx(start)
    assert fitted.end == pytest.approx(end)
    assert fitted.length == pytest.approx(segment.length)
    assert fitted.segment_id == 8
    assert fitted is not segment


@pytest.mark.parametrize(
    ("first", "second", "start", "end"),
    [
        pytest.param(
            ((1.0, 3.0), (5.0, 3.0)),
            ((8.0, 3.0), (12.0, 3.0)),
            (1.0, 3.0),
            (12.0, 3.0),
            id="horizontal",
        ),
        pytest.param(
            ((3.0, 1.0), (3.0, 5.0)),
            ((3.0, 8.0), (3.0, 12.0)),
            (3.0, 1.0),
            (3.0, 12.0),
            id="vertical",
        ),
        pytest.param(
            ((1.0, 3.0), (5.0, 7.0)),
            ((8.0, 10.0), (12.0, 14.0)),
            (1.0, 3.0),
            (12.0, 14.0),
            id="diagonal",
        ),
    ],
)
def test_collinear_fragments_fit_their_full_extent(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    group = SegmentGroup((LineSegment(0, *first), LineSegment(1, *second)))

    fitted = fit_group_line(group, segment_id=2)

    assert fitted.start == pytest.approx(start)
    assert fitted.end == pytest.approx(end)


def test_noisy_fit_uses_orthogonal_distances_and_extreme_projections() -> None:
    # Symmetric perpendicular noise around y=x+2 keeps the principal axis known.
    group = SegmentGroup(
        (
            LineSegment(0, (0.1, 1.9), (9.9, 12.1)),
            LineSegment(1, (-0.1, 2.1), (10.1, 11.9)),
        )
    )

    fitted = fit_group_line(group, segment_id=2)

    assert fitted.start == pytest.approx((0.0, 2.0))
    assert fitted.end == pytest.approx((10.0, 12.0))
    for segment in group.segments:
        for point in (segment.start, segment.end):
            assert point_line_distance(point, fitted) < 0.15


def test_fit_is_independent_of_order_direction_and_source_ids() -> None:
    segments = (
        LineSegment(40, (4.1, 0.0), (3.9, 10.0)),
        LineSegment(90, (4.05, 12.0), (4.0, 20.0)),
        LineSegment(10, (3.95, 22.0), (4.1, 30.0)),
    )
    expected = fit_group_line(SegmentGroup(segments), segment_id=100)

    for order in permutations(segments):
        for reversals in product((False, True), repeat=len(segments)):
            reordered = tuple(
                (
                    LineSegment(i, s.end, s.start)
                    if reverse
                    else LineSegment(i, s.start, s.end)
                )
                for i, (s, reverse) in enumerate(zip(order, reversals))
            )
            actual = fit_group_line(SegmentGroup(reordered), segment_id=101)

            assert actual.start == pytest.approx(expected.start)
            assert actual.end == pytest.approx(expected.end)
            assert actual.end[1] > actual.start[1]


def test_fitting_preserves_source_objects_and_does_not_fill_their_gap() -> None:
    first = LineSegment(0, (0.0, 3.0), (5.0, 3.0))
    second = LineSegment(1, (20.0, 3.0), (30.0, 3.0))
    sources = (first, second)
    group = SegmentGroup(sources)
    before = tuple((s.segment_id, s.start, s.end) for s in sources)

    fitted = fit_group_line(group, segment_id=2)

    assert fitted.start == pytest.approx((0.0, 3.0))
    assert fitted.end == pytest.approx((30.0, 3.0))
    assert fitted.length > sum(s.length for s in sources)
    assert group.segments is sources
    assert group.segments[0] is first
    assert group.segments[1] is second
    assert tuple((s.segment_id, s.start, s.end) for s in sources) == before
    assert projected_segment_gap(*group.segments) == 15.0


def test_rejects_empty_group() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        fit_group_line(SegmentGroup(()), segment_id=0)


@pytest.mark.parametrize("count", [1, 3])
def test_rejects_identical_endpoints(count: int) -> None:
    group = SegmentGroup(
        tuple(LineSegment(i, (2.0, 3.0), (2.0, 3.0)) for i in range(count))
    )

    with pytest.raises(ValueError, match="distinct points"):
        fit_group_line(group, segment_id=count)


def test_zero_length_fragment_can_contribute_to_a_non_degenerate_group() -> None:
    group = SegmentGroup(
        (LineSegment(0, (2.0, 3.0), (2.0, 3.0)), LineSegment(1, (5.0, 3.0), (9.0, 3.0)))
    )

    fitted = fit_group_line(group, segment_id=2)

    assert fitted.start == pytest.approx((2.0, 3.0))
    assert fitted.end == pytest.approx((9.0, 3.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("coordinate", range(4))
def test_rejects_nonfinite_source_coordinates(value: float, coordinate: int) -> None:
    coordinates = [1.0, 2.0, 10.0, 20.0]
    coordinates[coordinate] = value
    group = SegmentGroup(
        (
            LineSegment(
                0, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
            ),
        )
    )

    with pytest.raises(ValueError, match="finite coordinates"):
        fit_group_line(group, segment_id=1)


def test_rejects_ambiguous_principal_direction() -> None:
    group = SegmentGroup(
        (
            LineSegment(0, (-1.0, -1.0), (1.0, 1.0)),
            LineSegment(1, (-1.0, 1.0), (1.0, -1.0)),
        )
    )

    with pytest.raises(ValueError, match="no unique principal direction"):
        fit_group_line(group, segment_id=2)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_scatter_matrix_is_scaled_before_squaring(scale: float) -> None:
    group = SegmentGroup((LineSegment(0, (-scale, -scale), (scale, scale)),))

    fitted = fit_group_line(group, segment_id=1)

    assert all(isfinite(value) for value in (*fitted.start, *fitted.end))
    assert fitted.start[0] / scale == pytest.approx(-1.0)
    assert fitted.start[1] / scale == pytest.approx(-1.0)
    assert fitted.end[0] / scale == pytest.approx(1.0)
    assert fitted.end[1] / scale == pytest.approx(1.0)


@pytest.mark.parametrize("segment_id", [-1, 0, 5, 9])
def test_requires_id_above_group_source_ids(segment_id: int) -> None:
    group = SegmentGroup(
        (LineSegment(0, (0.0, 0.0), (1.0, 0.0)), LineSegment(9, (2.0, 0.0), (3.0, 0.0)))
    )

    with pytest.raises(ValueError, match="above source segment IDs"):
        fit_group_line(group, segment_id=segment_id)


def test_grouping_then_fitting_keeps_separate_edges_and_source_gaps() -> None:
    sources = (
        LineSegment(8, (12.0, 3.0), (20.0, 3.0)),
        LineSegment(2, (0.0, 3.0), (8.0, 3.0)),
        LineSegment(15, (0.0, 30.0), (20.0, 30.0)),
    )
    groups = group_segments(
        sources, angle_tolerance=3.0, distance_tolerance=2.0, gap_tolerance=12.0
    )
    first_id = max(s.segment_id for s in sources) + 1

    fitted = tuple(
        fit_group_line(g, segment_id=first_id + i) for i, g in enumerate(groups)
    )

    assert len(groups) == len(fitted) == 2
    assert fitted[0].start == pytest.approx((0.0, 3.0))
    assert fitted[0].end == pytest.approx((20.0, 3.0))
    assert fitted[1].start == pytest.approx((0.0, 30.0))
    assert fitted[1].end == pytest.approx((20.0, 30.0))
    assert projected_segment_gap(*groups[0].segments) == 4.0
    assert {line.segment_id for line in fitted}.isdisjoint(
        s.segment_id for s in sources
    )
    assert all(
        any(member is source for source in sources)
        for g in groups
        for member in g.segments
    )

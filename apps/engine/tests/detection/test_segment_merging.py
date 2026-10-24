from math import cos, radians, sin

import pytest

from parkvision.detection.line_segment import LineSegment
from parkvision.detection.segment_merging import (
    angle_difference,
    are_nearly_collinear,
    point_line_distance,
    segment_angle,
)


@pytest.mark.parametrize(
    ("start", "end", "expected_angle"),
    [
        pytest.param((2.0, 3.0), (6.0, 3.0), 0.0, id="horizontal"),
        pytest.param((2.0, 3.0), (2.0, 7.0), 90.0, id="vertical"),
        pytest.param((2.0, 3.0), (6.0, 7.0), 45.0, id="diagonal-45"),
        pytest.param((6.0, 3.0), (2.0, 7.0), 135.0, id="diagonal-135"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_segment_angle_is_independent_of_endpoint_order(
    start: tuple[float, float],
    end: tuple[float, float],
    expected_angle: float,
    reverse: bool,
) -> None:
    if reverse:
        start, end = end, start
    segment = LineSegment(segment_id=0, start=start, end=end)

    assert segment_angle(segment) == pytest.approx(expected_angle)


@pytest.mark.parametrize(
    ("first_angle", "second_angle", "expected_difference"),
    [
        pytest.param(45.0, 45.0, 0.0, id="identical-directions"),
        pytest.param(175.0, 5.0, 10.0, id="wrap-around-180"),
        pytest.param(0.0, 90.0, 90.0, id="perpendicular"),
    ],
)
@pytest.mark.parametrize("swap", [False, True], ids=["original-order", "swapped"])
def test_angle_difference_is_independent_of_argument_order(
    first_angle: float,
    second_angle: float,
    expected_difference: float,
    swap: bool,
) -> None:
    first = LineSegment(
        segment_id=0,
        start=(0.0, 0.0),
        end=(cos(radians(first_angle)), sin(radians(first_angle))),
    )
    second = LineSegment(
        segment_id=1,
        start=(0.0, 0.0),
        end=(cos(radians(second_angle)), sin(radians(second_angle))),
    )
    if swap:
        first, second = second, first

    assert angle_difference(first, second) == pytest.approx(expected_difference)


@pytest.mark.parametrize(
    ("point", "expected_distance"),
    [
        pytest.param((5.0, 10.0), 0.0, id="point-on-line"),
        pytest.param((5.0, 5.0), 5.0, id="point-five-pixels-above-line"),
        pytest.param((15.0, 10.0), 0.0, id="point-beyond-segment-end"),
    ],
)
def test_point_line_distance_measures_distance_to_infinite_line(
    point: tuple[float, float], expected_distance: float
) -> None:
    line = LineSegment(segment_id=0, start=(0.0, 10.0), end=(10.0, 10.0))

    assert point_line_distance(point, line) == pytest.approx(expected_distance)


def test_point_line_distance_rejects_zero_length_segment() -> None:
    line = LineSegment(segment_id=0, start=(2.0, 3.0), end=(2.0, 3.0))

    with pytest.raises(ValueError):
        point_line_distance((5.0, 8.0), line)


@pytest.mark.parametrize(
    (
        "second_start",
        "second_end",
        "angle_tolerance",
        "distance_tolerance",
        "expected",
    ),
    [
        pytest.param((20.0, 0.0), (30.0, 0.0), 0.0, 0.0, True, id="same-line-with-gap"),
        pytest.param(
            (0.0, 1.0), (10.0, 1.0), 0.0, 2.0, True, id="parallel-within-distance"
        ),
        pytest.param(
            (0.0, 2.0), (10.0, 2.0), 0.0, 2.0, True, id="distance-at-tolerance"
        ),
        pytest.param(
            (0.0, 3.0), (10.0, 3.0), 0.0, 2.0, False, id="distance-above-tolerance"
        ),
        pytest.param(
            (0.0, 0.0), (10.0, 10.0), 44.0, 100.0, False, id="angle-above-tolerance"
        ),
        pytest.param(
            (0.0, 0.0), (10.0, 10.0), 45.0, 100.0, True, id="angle-at-tolerance"
        ),
        pytest.param((0.0, 0.0), (10.0, 1.0), 6.0, 1.0, True, id="nearly-collinear"),
        pytest.param(
            (0.0, 1.0), (10.0, 1.0), 0.0, 0.0, False, id="zero-distance-tolerance"
        ),
        pytest.param(
            (0.0, 0.0), (10.0, 1.0), 0.0, 100.0, False, id="zero-angle-tolerance"
        ),
        pytest.param(
            (0.0, 0.0),
            (1.0, 1.0),
            45.0,
            2.0,
            False,
            id="short-segment-close-to-long-line-only",
        ),
    ],
)
@pytest.mark.parametrize(
    "reverse_first", [False, True], ids=["first", "first-reversed"]
)
@pytest.mark.parametrize(
    "reverse_second", [False, True], ids=["second", "second-reversed"]
)
@pytest.mark.parametrize("swap", [False, True], ids=["original-order", "swapped"])
def test_are_nearly_collinear_is_independent_of_endpoint_and_argument_order(
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    angle_tolerance: float,
    distance_tolerance: float,
    expected: bool,
    reverse_first: bool,
    reverse_second: bool,
    swap: bool,
) -> None:
    first_start, first_end = (0.0, 0.0), (10.0, 0.0)
    if reverse_first:
        first_start, first_end = first_end, first_start
    if reverse_second:
        second_start, second_end = second_end, second_start
    first = LineSegment(segment_id=0, start=first_start, end=first_end)
    second = LineSegment(segment_id=1, start=second_start, end=second_end)
    if swap:
        first, second = second, first

    assert (
        are_nearly_collinear(first, second, angle_tolerance, distance_tolerance)
        is expected
    )


@pytest.mark.parametrize(
    ("angle_tolerance", "distance_tolerance"),
    [
        pytest.param(-1.0, 1.0, id="negative-angle"),
        pytest.param(1.0, -1.0, id="negative-distance"),
        pytest.param(float("nan"), 1.0, id="nan-angle"),
        pytest.param(1.0, float("nan"), id="nan-distance"),
        pytest.param(float("inf"), 1.0, id="infinite-angle"),
        pytest.param(1.0, float("inf"), id="infinite-distance"),
    ],
)
def test_are_nearly_collinear_rejects_invalid_tolerances(
    angle_tolerance: float, distance_tolerance: float
) -> None:
    first = LineSegment(segment_id=0, start=(0.0, 0.0), end=(10.0, 0.0))
    second = LineSegment(segment_id=1, start=(0.0, 0.0), end=(0.0, 10.0))

    with pytest.raises(ValueError):
        are_nearly_collinear(first, second, angle_tolerance, distance_tolerance)

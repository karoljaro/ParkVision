from dataclasses import FrozenInstanceError
from math import inf, nextafter

import pytest

from parkvision.domain.geometry.point import Point
from parkvision.domain.geometry.polygon import Polygon


@pytest.fixture
def triangle_vertices() -> tuple[Point, ...]:
    return (Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0))


@pytest.mark.parametrize(
    ("vertices", "expected_area"),
    [
        pytest.param(
            (Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
            0.5,
            id="triangle",
        ),
        pytest.param(
            (Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.6)),
            0.12,
            id="translated-triangle",
        ),
        pytest.param(
            (
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(1.0, 1.0),
                Point(0.0, 1.0),
            ),
            1.0,
            id="unit-square",
        ),
        pytest.param(
            (
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(1.0, 1.0),
                Point(0.5, 0.5),
                Point(0.0, 1.0),
            ),
            0.75,
            id="concave-polygon",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["ccw", "cw"])
def test_area_for_known_shapes_in_both_winding_orders(
    vertices: tuple[Point, ...], expected_area: float, reverse: bool
) -> None:
    ordered_vertices = vertices[::-1] if reverse else vertices

    polygon = Polygon(ordered_vertices)

    assert polygon.vertices == ordered_vertices
    assert polygon.area == pytest.approx(expected_area)


@pytest.mark.parametrize("start", [0, 1, 2])
def test_area_does_not_depend_on_starting_vertex(start: int) -> None:
    vertices = (Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.6))

    polygon = Polygon(vertices[start:] + vertices[:start])

    assert polygon.area == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("vertices", "unique_count"),
    [
        pytest.param((), 0, id="empty"),
        pytest.param((Point(0.0, 0.0),), 1, id="one-vertex"),
        pytest.param((Point(0.0, 0.0), Point(1.0, 0.0)), 2, id="two-vertices"),
        pytest.param(
            (Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0)),
            1,
            id="three-equal-vertices",
        ),
        pytest.param(
            (
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(0.0, 0.0),
                Point(1.0, 0.0),
            ),
            2,
            id="two-unique-vertices-with-repetitions",
        ),
    ],
)
def test_requires_at_least_three_unique_vertices(
    vertices: tuple[Point, ...], unique_count: int
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"^A polygon must have at least 3 unique vertices, got {unique_count}\.$"
        ),
    ):
        Polygon(vertices)


@pytest.mark.parametrize(
    "vertices",
    [
        pytest.param(
            (Point(0.0, 0.5), Point(0.5, 0.5), Point(1.0, 0.5)),
            id="horizontal",
        ),
        pytest.param(
            (Point(0.5, 0.0), Point(0.5, 0.5), Point(0.5, 1.0)),
            id="vertical",
        ),
        pytest.param(
            (Point(0.0, 0.0), Point(0.5, 0.5), Point(1.0, 1.0)),
            id="diagonal",
        ),
    ],
)
def test_rejects_collinear_vertices(vertices: tuple[Point, ...]) -> None:
    with pytest.raises(ValueError, match="^A polygon must have a non-zero area,"):
        Polygon(vertices)


@pytest.mark.parametrize(
    "height",
    [
        pytest.param(nextafter(2e-12, 0.0), id="below-area-threshold"),
        pytest.param(2e-12, id="at-area-threshold"),
    ],
)
def test_rejects_area_at_or_below_threshold(height: float) -> None:
    vertices = (Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, height))

    with pytest.raises(ValueError, match="^A polygon must have a non-zero area,"):
        Polygon(vertices)


def test_accepts_area_just_above_threshold() -> None:
    height = nextafter(2e-12, inf)
    vertices = (Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, height))

    polygon = Polygon(vertices)

    assert polygon.area > 1e-12
    assert polygon.area == pytest.approx(height / 2.0, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("repeat_first_at_end", [False, True])
def test_rejects_repeated_vertices(
    triangle_vertices: tuple[Point, ...], repeat_first_at_end: bool
) -> None:
    if repeat_first_at_end:
        vertices = triangle_vertices + (triangle_vertices[0],)
    else:
        vertices = (triangle_vertices[0],) + triangle_vertices

    with pytest.raises(ValueError, match="Polygon vertices must be unique"):
        Polygon(vertices)


def test_vertices_are_immutable(triangle_vertices: tuple[Point, ...]) -> None:
    polygon = Polygon(triangle_vertices)

    with pytest.raises(FrozenInstanceError):
        setattr(polygon, "vertices", triangle_vertices[::-1])


def _points(coordinates: tuple[tuple[float, float], ...]) -> tuple[Point, ...]:
    return tuple(Point(x, y) for x, y in coordinates)


@pytest.mark.parametrize(
    "coordinates",
    [
        pytest.param(
            ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
            id="symmetric-crossing-zero-algebraic-area",
        ),
        pytest.param(
            ((0.125, 0.125), (0.875, 0.875), (0.125, 0.875), (0.625, 0.125)),
            id="off-center-diagonal-crossing-positive-area",
        ),
        pytest.param(
            ((0.0, 0.5), (1.0, 0.5), (0.75, 1.0), (0.75, 0.0)),
            id="horizontal-vertical-crossing-positive-area",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("start", [0, 1], ids=["internal-edges", "closing-edge"])
def test_rejects_crossing_edges(
    coordinates: tuple[tuple[float, float], ...], reverse: bool, start: int
) -> None:
    """Crossing edge interiors are invalid regardless of vertex ordering."""
    vertices = _points(coordinates)
    vertices = vertices[start:] + vertices[:start]
    if reverse:
        vertices = vertices[::-1]

    with pytest.raises(ValueError):
        Polygon(vertices)


@pytest.mark.parametrize(
    "coordinates",
    [
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.25, 0.0),
                (0.75, 0.0),
                (0.0, 1.0),
            ),
            id="non-adjacent-horizontal-overlap",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (0.0, 0.25),
                (0.0, 0.75),
                (1.0, 0.0),
            ),
            id="non-adjacent-vertical-overlap",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.25, 0.25),
                (0.75, 0.75),
                (1.0, 0.25),
            ),
            id="non-adjacent-diagonal-overlap",
        ),
        pytest.param(
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.0, 1.0)),
            id="adjacent-backtracking",
        ),
        pytest.param(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.0)),
            id="first-and-closing-edge-overlap",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_rejects_overlapping_edges(
    coordinates: tuple[tuple[float, float], ...], reverse: bool
) -> None:
    """Even adjacent edges may share only their connecting vertex."""
    vertices = _points(coordinates)
    if reverse:
        vertices = vertices[::-1]

    with pytest.raises(ValueError):
        Polygon(vertices)


@pytest.mark.parametrize(
    "coordinates",
    [
        pytest.param(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (0.0, 1.0)),
            id="vertex-on-non-adjacent-edge-interior",
        ),
        pytest.param(
            ((0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0)),
            id="vertex-on-closing-edge-interior",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_rejects_non_adjacent_edge_touches(
    coordinates: tuple[tuple[float, float], ...], reverse: bool
) -> None:
    """A vertex touching an unrelated edge invalidates a simple polygon."""
    vertices = _points(coordinates)
    if reverse:
        vertices = vertices[::-1]

    with pytest.raises(ValueError):
        Polygon(vertices)


@pytest.mark.parametrize(
    "coordinates",
    [
        pytest.param(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            id="consecutive-duplicate-interior-vertex",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ),
            id="first-vertex-repeated-at-end",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ),
            id="multiple-duplicates-at-end",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.5),
                (0.0, 0.0),
                (0.5, 1.0),
                (0.0, 1.0),
            ),
            id="first-vertex-revisited-before-end",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (1.0, 0.5),
                (1.0, 1.0),
                (0.5, 0.5),
                (0.0, 1.0),
            ),
            id="non-adjacent-edges-share-repeated-vertex",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_rejects_repeated_vertices_in_canonical_representation(
    coordinates: tuple[tuple[float, float], ...], reverse: bool
) -> None:
    """Implicit closure requires every supplied vertex to appear exactly once."""
    vertices = _points(coordinates)
    if reverse:
        vertices = vertices[::-1]

    with pytest.raises(ValueError):
        Polygon(vertices)


@pytest.mark.parametrize(
    ("coordinates", "expected_area"),
    [
        pytest.param(
            (
                (0.25, 0.0),
                (0.75, 0.0),
                (1.0, 0.5),
                (0.75, 1.0),
                (0.25, 1.0),
                (0.0, 0.5),
            ),
            0.75,
            id="convex-hexagon",
        ),
        pytest.param(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.75, 1.0),
                (0.75, 0.25),
                (0.25, 0.25),
                (0.25, 1.0),
                (0.0, 1.0),
            ),
            0.625,
            id="concave-u-with-disjoint-collinear-edges",
        ),
        pytest.param(
            ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            1.0,
            id="adjacent-collinear-edges-share-only-vertex",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["ccw", "cw"])
def test_accepts_simple_polygons_with_valid_edge_contacts(
    coordinates: tuple[tuple[float, float], ...], expected_area: float, reverse: bool
) -> None:
    vertices = _points(coordinates)
    if reverse:
        vertices = vertices[::-1]

    polygon = Polygon(vertices)

    assert polygon.vertices == vertices
    assert polygon.area == pytest.approx(expected_area)

from dataclasses import FrozenInstanceError
from math import inf, nan, nextafter

import pytest

from parkvision.domain.geometry.point import Point


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("y", [0.0, 0.5, 1.0])
def test_accepts_coordinates_in_closed_unit_interval(x: float, y: float) -> None:
    point = Point(x, y)

    assert point.x == x
    assert point.y == y


@pytest.mark.parametrize("coordinate", ["x", "y"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-0.5, id="below-zero"),
        pytest.param(1.5, id="above-one"),
        pytest.param(nextafter(0.0, -inf), id="just-below-zero"),
        pytest.param(nextafter(1.0, inf), id="just-above-one"),
        pytest.param(inf, id="positive-infinity"),
        pytest.param(-inf, id="negative-infinity"),
        pytest.param(nan, id="nan"),
    ],
)
def test_rejects_invalid_coordinate(coordinate: str, value: float) -> None:
    coordinates = {"x": 0.5, "y": 0.5}
    coordinates[coordinate] = value

    with pytest.raises(
        ValueError, match=rf"^{coordinate} must be a finite number between 0 and 1,"
    ):
        Point(**coordinates)


@pytest.mark.parametrize("coordinate", ["x", "y"])
def test_coordinates_are_immutable(coordinate: str) -> None:
    point = Point(0.25, 0.75)

    with pytest.raises(FrozenInstanceError):
        setattr(point, coordinate, 0.5)


def test_equality_and_hash_depend_on_both_coordinates() -> None:
    point = Point(0.25, 0.75)
    same_point = Point(0.25, 0.75)
    different_x = Point(0.5, 0.75)
    different_y = Point(0.25, 0.5)

    assert point == same_point
    assert hash(point) == hash(same_point)
    assert point != different_x
    assert point != different_y
    assert len({point, same_point, different_x, different_y}) == 3

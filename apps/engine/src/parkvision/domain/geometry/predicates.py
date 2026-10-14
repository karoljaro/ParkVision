from typing import Final

from .point import Point

EPSILON: Final[float] = 1e-12


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        abs(orientation(a, b, c)) <= EPSILON
        and min(a.x, b.x) <= c.x <= max(a.x, b.x)
        and min(a.y, b.y) <= c.y <= max(a.y, b.y)
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) <= EPSILON and point_on_segment(a, b, c):
        return True
    if abs(o2) <= EPSILON and point_on_segment(a, b, d):
        return True
    if abs(o3) <= EPSILON and point_on_segment(c, d, a):
        return True
    if abs(o4) <= EPSILON and point_on_segment(c, d, b):
        return True

    return False

from math import atan2, degrees

from .line_segment import LineSegment

type PixelPoint = tuple[float, float]


def segment_angle(segment: LineSegment) -> float:
    """Return the angle of a line segment in degrees in the range [0, 180)."""
    x1, y1 = segment.start
    x2, y2 = segment.end

    return degrees(atan2(y2 - y1, x2 - x1)) % 180


def angle_difference(first: LineSegment, second: LineSegment) -> float:
    """Return the smallest angle difference between two line segments in degrees."""
    difference = abs(segment_angle(first) - segment_angle(second))

    return min(difference, 180 - difference)


def point_line_distance(point: PixelPoint, line: LineSegment) -> float:
    """Return the perpendicular distance from a point to an infinite line."""
    if line.length == 0:
        raise ValueError("Line segment must have a non-zero length.")

    px, py = point
    ax, ay = line.start
    bx, by = line.end

    numerator = abs((bx - ax) * (py - ay) - (by - ay) * (px - ax))

    return numerator / line.length

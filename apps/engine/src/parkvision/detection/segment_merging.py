from math import atan2, degrees

from .line_segment import LineSegment


def segment_angle(segment: LineSegment) -> float:
    """Return the angle of a line segment in degrees in the range [0, 180)."""
    x1, y1 = segment.start
    x2, y2 = segment.end

    return degrees(atan2(y2 - y1, x2 - x1)) % 180


def angle_difference(first: LineSegment, second: LineSegment) -> float:
    """Return the smallest angle difference between two line segments in degrees."""
    difference = abs(segment_angle(first) - segment_angle(second))

    return min(difference, 180 - difference)

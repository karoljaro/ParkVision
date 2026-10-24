from math import atan2, degrees, isfinite

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


def are_nearly_collinear(
    first: LineSegment,
    second: LineSegment,
    angle_tolerance: float,
    distance_tolerance: float,
) -> bool:
    """
    Determine if two line segments are nearly collinear within the given tolerances.
    """
    if not isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("Angle tolerance must be finite and non-negative.")

    if not isfinite(distance_tolerance) or distance_tolerance < 0:
        raise ValueError("Distance tolerance must be finite and non-negative.")

    if angle_difference(first, second) > angle_tolerance:
        return False

    if point_line_distance(first.start, second) > distance_tolerance:
        return False
    if point_line_distance(first.end, second) > distance_tolerance:
        return False
    if point_line_distance(second.start, first) > distance_tolerance:
        return False
    if point_line_distance(second.end, first) > distance_tolerance:
        return False

    return True


def projected_segment_gap(first: LineSegment, second: LineSegment) -> float:
    """Return the gap between segment projections along the first segment."""
    if first.length <= 0 or second.length <= 0:
        raise ValueError("Both segments must have a non-zero length.")

    ux = (first.end[0] - first.start[0]) / first.length
    uy = (first.end[1] - first.start[1]) / first.length

    t_first_min = 0.0
    t_first_max = first.length

    v_start_x = second.start[0] - first.start[0]
    v_start_y = second.start[1] - first.start[1]

    v_end_x = second.end[0] - first.start[0]
    v_end_y = second.end[1] - first.start[1]

    t_second_start = v_start_x * ux + v_start_y * uy
    t_second_end = v_end_x * ux + v_end_y * uy

    t_second_min = min(t_second_start, t_second_end)
    t_second_max = max(t_second_start, t_second_end)

    return max(0.0, t_second_min - t_first_max, t_first_min - t_second_max)

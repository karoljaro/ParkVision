from math import atan2, ceil, degrees, isfinite

from .line_segment import LineSegment
from .segment_group import SegmentGroup

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


def can_merge_segments(
    first: LineSegment,
    second: LineSegment,
    angle_tolerance: float,
    distance_tolerance: float,
    gap_tolerance: float,
) -> bool:
    """Check whether two line segments can be merged."""
    if not isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("Gap tolerance must be finite and non-negative.")

    if not are_nearly_collinear(first, second, angle_tolerance, distance_tolerance):
        return False

    if projected_segment_gap(first, second) > gap_tolerance:
        return False

    return True


def can_join_group(
    segment: LineSegment,
    group: SegmentGroup,
    angle_tolerance: float,
    distance_tolerance: float,
    gap_tolerance: float,
) -> bool:
    """Check whether a segment can join an existing group."""
    if not group.segments:
        raise ValueError("Segment group must contain at least one segment.")

    if not isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("Gap tolerance must be finite and non-negative.")

    if not all(
        are_nearly_collinear(
            existing,
            segment,
            angle_tolerance,
            distance_tolerance,
        )
        for existing in group.segments
    ):
        return False

    return any(
        projected_segment_gap(existing, segment) <= gap_tolerance
        for existing in group.segments
    )


def can_merge_groups(
    first: SegmentGroup,
    second: SegmentGroup,
    angle_tolerance: float,
    distance_tolerance: float,
    gap_tolerance: float,
) -> bool:
    """Check whether two segment groups can be merged."""
    if not first.segments or not second.segments:
        raise ValueError("Both segment groups must contain at least one segment.")

    if not isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("Gap tolerance must be finite and non-negative.")

    if not all(
        are_nearly_collinear(
            first_segment,
            second_segment,
            angle_tolerance,
            distance_tolerance,
        )
        for first_segment in first.segments
        for second_segment in second.segments
    ):
        return False

    return any(
        projected_segment_gap(first_segment, second_segment) <= gap_tolerance
        for first_segment in first.segments
        for second_segment in second.segments
    )


def group_segments(
    segments: tuple[LineSegment, ...],
    angle_tolerance: float,
    distance_tolerance: float,
    gap_tolerance: float,
) -> tuple[SegmentGroup, ...]:
    """Preserve ID-ordered greedy merges using cached pair relations.

    Bit masks encode original input positions, never segment IDs. Each group's
    common collinear mask is an intersection (all cross pairs must pass), while
    its close mask is a union (one directed projected gap must pass). Only
    relations involving the surviving group change after a merge.
    """
    if not isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("Angle tolerance must be finite and non-negative.")

    if not isfinite(distance_tolerance) or distance_tolerance < 0:
        raise ValueError("Distance tolerance must be finite and non-negative.")

    if not isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("Gap tolerance must be finite and non-negative.")

    ordered = sorted(segments, key=lambda segment: segment.segment_id)
    # Eager pair checks can raise on a degenerate pair that the greedy scan
    # would never reach. Such inputs use lazy candidates in the same traversal.
    use_pair_index = all(
        isfinite(length := segment.length)
        and length > 0
        and all(isfinite(value) for value in (*segment.start, *segment.end))
        for segment in ordered
    )

    count = len(ordered)
    bits = [1 << index for index in range(count)]
    members = bits.copy()
    common = [0] * count
    close = [0] * count
    if use_pair_index:
        # Cache the canonical angle result only for candidate selection. The
        # extra bucket on either side makes the index conservative at boundaries;
        # acceptance always goes through the original geometric functions.
        angle_bins = [int(segment_angle(segment)) % 180 for segment in ordered]
        buckets = [0] * 180
        for index, bucket in enumerate(angle_bins):
            buckets[bucket] |= bits[index]
        radius = min(90, ceil(min(angle_tolerance, 90)) + 1)
        angle_candidates = [0] * 180
        for bucket in range(180):
            for offset in range(-radius, radius + 1):
                angle_candidates[bucket] |= buckets[(bucket + offset) % 180]

        for first_index, first in enumerate(ordered):
            remaining = angle_candidates[angle_bins[first_index]] & ~(
                (bits[first_index] << 1) - 1
            )
            while remaining:
                second_bit = remaining & -remaining
                remaining ^= second_bit
                second_index = second_bit.bit_length() - 1
                second = ordered[second_index]
                if not are_nearly_collinear(
                    first, second, angle_tolerance, distance_tolerance
                ):
                    continue
                common[first_index] |= bits[second_index]
                common[second_index] |= bits[first_index]
                if projected_segment_gap(first, second) <= gap_tolerance:
                    close[first_index] |= bits[second_index]
                # Projection is directional; groups can interleave input IDs.
                if projected_segment_gap(second, first) <= gap_tolerance:
                    close[second_index] |= bits[first_index]

    active = (1 << count) - 1
    candidates = [
        (close[index] if use_pair_index else active) & ~((bits[index] << 1) - 1)
        for index in range(count)
    ]
    groups: list[tuple[LineSegment, ...]] = [(segment,) for segment in ordered]

    while True:
        # Stable root indices have exactly the old list's lexicographic order.
        for first_index in range(count):
            if active & bits[first_index] and candidates[first_index] & active:
                break
        else:
            return tuple(
                SegmentGroup(groups[index])
                for index in range(count)
                if active & bits[index]
            )

        remaining = candidates[first_index] & active
        second_bit = remaining & -remaining
        second_index = second_bit.bit_length() - 1
        if not use_pair_index and not can_merge_groups(
            SegmentGroup(groups[first_index]),
            SegmentGroup(groups[second_index]),
            angle_tolerance,
            distance_tolerance,
            gap_tolerance,
        ):
            candidates[first_index] &= ~second_bit
            continue

        groups[first_index] += groups[second_index]
        groups[second_index] = ()
        members[first_index] |= members[second_index]
        common[first_index] &= common[second_index]
        close[first_index] |= close[second_index]
        active ^= second_bit

        candidates[first_index] = 0
        for index in range(count):
            if index == first_index or not active & bits[index]:
                continue
            lower, upper = (
                (first_index, index) if first_index < index else (index, first_index)
            )
            if not use_pair_index or (
                common[lower] & members[upper] == members[upper]
                and close[lower] & members[upper]
            ):
                candidates[lower] |= bits[upper]
            else:
                candidates[lower] &= ~bits[upper]

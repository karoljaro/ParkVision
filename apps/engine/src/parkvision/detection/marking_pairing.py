"""Geometric hypotheses for the two visible edges of a painted marking."""

from math import hypot, isfinite

from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .segment_merging import PixelPoint, angle_difference


def pair_marking_edges(
    fitted_lines: tuple[LineSegment, ...],
    *,
    min_width_px: float,
    max_width_px: float,
    angle_tolerance: float = 3.0,
    min_overlap_px: float = 30.0,
    min_overlap_ratio: float = 0.6,
    max_width_ratio: float = 1.5,
) -> tuple[MarkingPair, ...]:
    """Pair nearby edges using a deterministic, disjoint greedy matching.

    Align undirected edge orientations, then project onto their mean direction.
    Measure separation at BOTH ends of the shared range, without extrapolating
    either edge. Reject crossings, out-of-range widths and excessive taper.
    Mild perspective is allowed; width bounds apply in the input image's pixels,
    not in world units. The positive minimum width rejects coincident edges.

    Rank valid candidates by decreasing overlap ratio, increasing mean width,
    increasing angle difference, decreasing shared length, then edge IDs. Take
    each candidate only if neither edge has been used. This is a local heuristic,
    not maximum-weight matching; equal geometric evidence is resolved by IDs.
    A long edge can pair only once, even if its partner is split into groups.

    Preserve all input objects, including unpaired lines. Extents can span source
    gaps; this function cannot infer paint continuity, LSD contrast polarity or
    distinguish nearby curbs/shadows from paint using geometry alone.

    Centerline IDs are allocated above all input IDs in candidate order. They
    are local to this call, may have gaps and do not encode source provenance.
    Invalid parameters, nonfinite/zero-length lines or duplicate IDs raise
    ValueError, even for a singleton input.
    """
    _validate_inputs(
        fitted_lines,
        min_width_px=min_width_px,
        max_width_px=max_width_px,
        angle_tolerance=angle_tolerance,
        min_overlap_px=min_overlap_px,
        min_overlap_ratio=min_overlap_ratio,
        max_width_ratio=max_width_ratio,
    )

    # Short lines cannot satisfy the absolute overlap. Keep them in the input,
    # but avoid quadratic candidate work on the numerous tiny LSD fragments.
    eligible_edges = sorted(
        (line for line in fitted_lines if line.length >= min_overlap_px),
        key=lambda line: line.segment_id,
    )
    first_centerline_id = (
        max((line.segment_id for line in fitted_lines), default=-1) + 1
    )
    candidates: list[MarkingPair] = []
    for first_index, first in enumerate(eligible_edges):
        for second in eligible_edges[first_index + 1 :]:
            if angle_difference(first, second) > angle_tolerance:
                continue

            candidate = _pair_candidate(
                first,
                second,
                centerline_id=first_centerline_id + len(candidates),
                min_width_px=min_width_px,
                max_width_px=max_width_px,
                min_overlap_px=min_overlap_px,
                min_overlap_ratio=min_overlap_ratio,
                max_width_ratio=max_width_ratio,
            )
            if candidate is not None:
                candidates.append(candidate)

    return _select_disjoint_pairs(candidates)


def _pair_candidate(
    first: LineSegment,
    second: LineSegment,
    *,
    centerline_id: int,
    min_width_px: float,
    max_width_px: float,
    min_overlap_px: float,
    min_overlap_ratio: float,
    max_width_ratio: float,
) -> MarkingPair | None:
    """Check shared coverage and width, then build the candidate's centerline."""
    # 1. Describe both edges in the same (along, across) coordinate system.
    origin = first.start
    direction = _mean_direction(first, second)
    first_start, first_end = _project_edge(first, origin, direction)
    second_start, second_end = _project_edge(second, origin, direction)

    # 2. Use only the interval supported by both fitted extents.
    overlap_start = max(first_start[0], second_start[0])
    overlap_end = min(first_end[0], second_end[0])
    overlap_length = overlap_end - overlap_start
    if overlap_length < min_overlap_px:
        return None

    first_extent = first_end[0] - first_start[0]
    second_extent = second_end[0] - second_start[0]
    overlap_ratio = overlap_length / max(first_extent, second_extent)
    if overlap_ratio < min_overlap_ratio:
        return None

    # 3. Measure both cross-sections; a sign change means the edges cross.
    first_across_start = _transverse_position(first_start, first_end, overlap_start)
    first_across_end = _transverse_position(first_start, first_end, overlap_end)
    second_across_start = _transverse_position(second_start, second_end, overlap_start)
    second_across_end = _transverse_position(second_start, second_end, overlap_end)

    start_separation = second_across_start - first_across_start
    end_separation = second_across_end - first_across_end
    if (start_separation > 0) != (end_separation > 0):
        return None

    widths = (abs(start_separation), abs(end_separation))
    narrower_width = min(widths)
    wider_width = max(widths)
    if not min_width_px <= narrower_width <= wider_width <= max_width_px:
        return None
    if wider_width / narrower_width > max_width_ratio:
        return None

    # 4. Take each cross-section's midpoint and return to image coordinates.
    center_across_start = first_across_start / 2 + second_across_start / 2
    center_across_end = first_across_end / 2 + second_across_end / 2
    centerline_start = _image_point(
        overlap_start, center_across_start, origin, direction
    )
    centerline_end = _image_point(overlap_end, center_across_end, origin, direction)
    endpoints = (*centerline_start, *centerline_end)
    if (
        not all(isfinite(value) for value in endpoints)
        or centerline_start == centerline_end
    ):
        raise ValueError("Centerline endpoints must represent a finite line.")

    return MarkingPair(
        edges=(first, second),
        centerline=LineSegment(centerline_id, centerline_start, centerline_end),
        widths_px=widths,
        overlap_ratio=overlap_ratio,
    )


def _mean_direction(first: LineSegment, second: LineSegment) -> PixelPoint:
    """Align unit vectors, normalize their sum and choose a stable sign."""
    first_dx = (first.end[0] - first.start[0]) / first.length
    first_dy = (first.end[1] - first.start[1]) / first.length
    second_dx = (second.end[0] - second.start[0]) / second.length
    second_dy = (second.end[1] - second.start[1]) / second.length

    dot_product = first_dx * second_dx + first_dy * second_dy
    if dot_product < 0:
        second_dx, second_dy = -second_dx, -second_dy

    summed_length = hypot(first_dx + second_dx, first_dy + second_dy)
    direction_x = (first_dx + second_dx) / summed_length
    direction_y = (first_dy + second_dy) / summed_length

    dominant_component = (
        direction_x if abs(direction_x) >= abs(direction_y) else direction_y
    )
    if dominant_component < 0:
        direction_x, direction_y = -direction_x, -direction_y
    return direction_x, direction_y


def _project_edge(
    edge: LineSegment, origin: PixelPoint, direction: PixelPoint
) -> tuple[PixelPoint, PixelPoint]:
    """Return endpoints as (along, across), ordered by increasing along value."""
    direction_x, direction_y = direction
    normal_x, normal_y = -direction_y, direction_x

    def project(point: PixelPoint) -> PixelPoint:
        offset_x = point[0] - origin[0]
        offset_y = point[1] - origin[1]
        along = offset_x * direction_x + offset_y * direction_y
        across = offset_x * normal_x + offset_y * normal_y
        return along, across

    start, end = sorted((project(edge.start), project(edge.end)))
    if not all(isfinite(value) for point in (start, end) for value in point):
        raise ValueError("Projected geometry must be finite.")
    return start, end


def _transverse_position(start: PixelPoint, end: PixelPoint, along: float) -> float:
    """Interpolate across = q_start + fraction * (q_end - q_start)."""
    fraction = (along - start[0]) / (end[0] - start[0])
    return start[1] + fraction * (end[1] - start[1])


def _image_point(
    along: float, across: float, origin: PixelPoint, direction: PixelPoint
) -> PixelPoint:
    """Invert projection: point = origin + along * direction + across * normal."""
    direction_x, direction_y = direction
    normal_x, normal_y = -direction_y, direction_x
    return (
        origin[0] + along * direction_x + across * normal_x,
        origin[1] + along * direction_y + across * normal_y,
    )


def _candidate_priority(
    pair: MarkingPair,
) -> tuple[float, float, float, float, int, int]:
    """Compare criteria in order; negative values put larger measurements first."""
    mean_width = sum(pair.widths_px) / 2
    return (
        -pair.overlap_ratio,
        mean_width,
        angle_difference(*pair.edges),
        -pair.centerline.length,
        pair.edges[0].segment_id,
        pair.edges[1].segment_id,
    )


def _select_disjoint_pairs(candidates: list[MarkingPair]) -> tuple[MarkingPair, ...]:
    """Take candidates in priority order, allowing each edge only one partner."""
    used_edge_ids: set[int] = set()
    selected_pairs: list[MarkingPair] = []
    for candidate in sorted(candidates, key=_candidate_priority):
        candidate_edge_ids = {edge.segment_id for edge in candidate.edges}
        if used_edge_ids.intersection(candidate_edge_ids):
            continue
        selected_pairs.append(candidate)
        used_edge_ids.update(candidate_edge_ids)
    return tuple(selected_pairs)


def _validate_inputs(
    fitted_lines: tuple[LineSegment, ...],
    *,
    min_width_px: float,
    max_width_px: float,
    angle_tolerance: float,
    min_overlap_px: float,
    min_overlap_ratio: float,
    max_width_ratio: float,
) -> None:
    """Reject invalid parameters and edges before filtering or comparing them."""
    if not (
        isfinite(min_width_px)
        and isfinite(max_width_px)
        and 0 < min_width_px <= max_width_px
    ):
        raise ValueError("Width bounds must be finite and 0 < min <= max.")
    if not isfinite(angle_tolerance) or not 0 <= angle_tolerance < 90:
        raise ValueError("Angle tolerance must be finite and in [0, 90).")
    if not isfinite(min_overlap_px) or min_overlap_px <= 0:
        raise ValueError("Minimum overlap must be finite and positive.")
    if not isfinite(min_overlap_ratio) or not 0 < min_overlap_ratio <= 1:
        raise ValueError("Minimum overlap ratio must be finite and in (0, 1].")
    if not isfinite(max_width_ratio) or max_width_ratio < 1:
        raise ValueError("Maximum width ratio must be finite and at least 1.")

    if len({line.segment_id for line in fitted_lines}) != len(fitted_lines):
        raise ValueError("Fitted edge IDs must be unique.")
    for line in fitted_lines:
        has_finite_endpoints = all(
            isfinite(value) for value in (*line.start, *line.end)
        )
        has_valid_length = isfinite(line.length) and line.length > 0
        if not has_finite_endpoints or not has_valid_length:
            raise ValueError("Fitted edges must have finite, non-zero geometry.")

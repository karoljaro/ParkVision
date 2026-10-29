"""Conservative two-line topology hypotheses in subpixel image coordinates."""

from collections import defaultdict
from math import floor, hypot, isfinite
from typing import Iterator, Literal

from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .segment_merging import PixelPoint, angle_difference
from .topology_candidate import (
    EndpointCandidate,
    JunctionCandidate,
    LineIntersection,
    TopologyCandidates,
)
from .topology_geometry import TopologyGeometry

# TODO: CREATE CONSTANT FOR THESE OPTIONS, sepcially for epsilons

_PIXEL_EPSILON = 1e-9
_PARALLEL_EPSILON = 1e-12
_MIN_GRID_CELL_PX = 64.0


def find_topology_candidates(
    pairs: tuple[MarkingPair, ...],
    unpaired_lines: tuple[LineSegment, ...] = (),
    *,
    endpoint_tolerance_px: float = 3.0,
    max_extension_px: float = 0.0,
    min_crossing_angle: float = 15.0,
    dedup_radius_px: float = 3.0,
    min_context_length_px: float = 30.0,
) -> TopologyCandidates:
    """Intersect axes, with optional unmatched edges as weaker endpoint context.

    L = end/end, T = end/interior, X = interior/interior. A line has two arms
    only when the hit is farther than endpoint_tolerance_px from BOTH ends;
    one when only one arm exceeds that length. Skip a hit if either line has
    no resolvable arm. These are pairwise geometric labels, not paint tests.
    Endpoint tolerance NEVER extends a segment. Any extension is explicitly
    bounded per line by max_extension_px and recorded as hypothetical (default
    zero disables it). Only 1e-9 px numerical roundoff is snapped to an endpoint.

    Unpaired edges are not paint: never intersect context with context, require
    min_context_length_px for participation and accept axis/context hits only at
    the axis END. Short/isolated/context lines and all endpoints remain intact.
    Source groups stay with the caller: extents and axes may bridge LSD gaps.

    Use a spatial bbox grid before angle and intersection tests. Only axes query
    the grid. Cell size adapts to long lines/extension, bounding cells per bbox;
    dense overlapping geometry can still have quadratic output/work. Deduplicate
    within a fixed anchor radius, without transitive chaining; prefer extent
    hits over extrapolated ones, then axis-only evidence. Keep all exact hits.
    Output is deterministic by geometry/IDs, independent of input order.

    Reject nonfinite/zero-length geometry, ID collisions (including pair edges),
    or invalid tolerances before processing even empty/singleton inputs.
    """
    # 1. Reject invalid thresholds, geometry and IDs before building the search index.
    _validate_parameters(
        endpoint_tolerance_px=endpoint_tolerance_px,
        max_extension_px=max_extension_px,
        min_crossing_angle=min_crossing_angle,
        dedup_radius_px=dedup_radius_px,
        min_context_length_px=min_context_length_px,
    )
    _validate_source_lines(pairs, unpaired_lines)

    # 2. Keep every axis/context line; cached geometry never replaces source objects.
    axes = tuple(pair.centerline for pair in pairs)
    axis_ids = {line.segment_id for line in axes}
    lines = tuple(sorted(axes + unpaired_lines, key=lambda line: line.segment_id))
    geometry = {line.segment_id: _geometry(line, max_extension_px) for line in lines}

    # 3. Find bounded pairwise hits, then group nearby hits without losing evidence.
    intersections = _find_intersections(
        geometry,
        axis_ids,
        endpoint_tolerance_px=endpoint_tolerance_px,
        max_extension_px=max_extension_px,
        min_crossing_angle=min_crossing_angle,
        min_context_length_px=min_context_length_px,
    )
    junctions = _deduplicate(intersections, dedup_radius_px)
    # 4. Retain all lines and both endpoints, including those without any junction.
    return TopologyCandidates(
        lines=lines,
        unpaired_line_ids=tuple(sorted(line.segment_id for line in unpaired_lines)),
        junctions=junctions,
        endpoints=_collect_endpoints(lines, junctions),
    )


def _validate_parameters(
    *,
    endpoint_tolerance_px: float,
    max_extension_px: float,
    min_crossing_angle: float,
    dedup_radius_px: float,
    min_context_length_px: float,
) -> None:
    """Validate thresholds before looking at geometry, including empty input."""
    for name, value in (
        ("endpoint_tolerance_px", endpoint_tolerance_px),
        ("max_extension_px", max_extension_px),
        ("dedup_radius_px", dedup_radius_px),
        ("min_context_length_px", min_context_length_px),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not isfinite(min_crossing_angle) or not 0 < min_crossing_angle <= 90:
        raise ValueError("min_crossing_angle must be finite and in (0, 90].")


def _validate_source_lines(
    pairs: tuple[MarkingPair, ...], unpaired_lines: tuple[LineSegment, ...]
) -> None:
    """Check all IDs and source geometry, including edges not used in searches."""
    all_lines = (
        tuple(line for pair in pairs for line in (*pair.edges, pair.centerline))
        + unpaired_lines
    )
    if len({line.segment_id for line in all_lines}) != len(all_lines):
        raise ValueError("Axis, paired edge and unpaired IDs must be unique.")
    for line in all_lines:
        if not (
            all(isfinite(v) for v in (*line.start, *line.end))
            and isfinite(line.length)
            and line.length > 0
        ):
            raise ValueError("Lines must have finite, non-zero geometry.")


def _find_intersections(
    geometry: dict[int, TopologyGeometry],
    axis_ids: set[int],
    *,
    endpoint_tolerance_px: float,
    max_extension_px: float,
    min_crossing_angle: float,
    min_context_length_px: float,
) -> list[LineIntersection]:
    """Run exact geometry only on nearby, sufficiently different directions."""
    intersections = []
    for first, second in _nearby_line_pairs(
        geometry, axis_ids, max_extension_px, min_context_length_px
    ):
        # Nearly parallel directions give unstable or distant line intersections.
        if angle_difference(first.line, second.line) < min_crossing_angle:
            continue
        intersection = _intersection(
            first,
            second,
            endpoint_tolerance_px,
            max_extension_px,
            uses_context=second.line.segment_id not in axis_ids,
        )
        if intersection is not None:
            intersections.append(intersection)
    return intersections


def _nearby_line_pairs(
    geometry: dict[int, TopologyGeometry],
    axis_ids: set[int],
    max_extension_px: float,
    min_context_length_px: float,
) -> Iterator[tuple[TopologyGeometry, TopologyGeometry]]:
    """Yield each nearby axis/axis or axis/context pair once; never context/context."""
    # 1. Exclude short context from searching only; it stays in the final result.
    eligible = tuple(
        item
        for line_id, item in geometry.items()
        if line_id in axis_ids or item.length >= min_context_length_px
    )
    # 2. Scale the grid for long lines so one bbox does not occupy excessive cells.
    cell_size = max(
        _MIN_GRID_CELL_PX,
        max_extension_px,
        max((g.length for g in eligible), default=0.0) / 32,
    )
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in eligible:
        for cell in _cells(item.bbox, cell_size):
            grid[cell].append(item.line.segment_id)

    # 3. Only axes query the index; ID ordering removes duplicate axis/axis pairs.
    for axis_id in sorted(axis_ids):
        first = geometry[axis_id]
        neighbors = {
            line_id
            for cell in _cells(first.bbox, cell_size)
            for line_id in grid.get(cell, ())
            if line_id != axis_id and (line_id not in axis_ids or line_id > axis_id)
        }
        for other_id in sorted(neighbors):
            second = geometry[other_id]
            if _overlap(first.bbox, second.bbox):
                yield first, second


def _collect_endpoints(
    lines: tuple[LineSegment, ...], junctions: tuple[JunctionCandidate, ...]
) -> tuple[EndpointCandidate, ...]:
    """Keep both ends of every line and link end-role hits to the nearest end."""
    line_by_id = {line.segment_id: line for line in lines}
    endpoint_nodes: dict[tuple[int, str], set[int]] = defaultdict(set)
    end_names: tuple[Literal["start"], Literal["end"]] = ("start", "end")
    for node in junctions:
        for intersection in node.intersections:
            for line_id, role in zip(intersection.line_ids, intersection.roles):
                if role == "end":
                    line = line_by_id[line_id]
                    end = min(
                        end_names,
                        key=lambda end: hypot(
                            getattr(line, end)[0] - intersection.position[0],
                            getattr(line, end)[1] - intersection.position[1],
                        ),
                    )
                    endpoint_nodes[line_id, end].add(node.candidate_id)
    # An unmatched endpoint is still useful evidence, so always emit both ends.
    return tuple(
        EndpointCandidate(
            line.segment_id,
            end,
            getattr(line, end),
            tuple(sorted(endpoint_nodes[line.segment_id, end])),
        )
        for line in lines
        for end in end_names
    )


def _geometry(line: LineSegment, extension: float) -> TopologyGeometry:
    """Cache unit direction and a search bbox expanded by the explicit pixel limit."""
    length = line.length
    # A unit direction makes distances along the line directly measurable in pixels.
    ux = (line.end[0] - line.start[0]) / length
    uy = (line.end[1] - line.start[1]) / length
    # Expand the search extent only; the stored line keeps its original endpoints.
    start = (line.start[0] - ux * extension, line.start[1] - uy * extension)
    end = (line.end[0] + ux * extension, line.end[1] + uy * extension)
    if not all(isfinite(v) for v in (*start, *end)):
        raise ValueError("Extended geometry must be finite.")
    # Pad for arithmetic roundoff only, not endpoint classification tolerance.
    bbox = (
        min(start[0], end[0]) - _PIXEL_EPSILON,
        min(start[1], end[1]) - _PIXEL_EPSILON,
        max(start[0], end[0]) + _PIXEL_EPSILON,
        max(start[1], end[1]) + _PIXEL_EPSILON,
    )
    return TopologyGeometry(line, (ux, uy), length, bbox)


def _cells(
    bbox: tuple[float, float, float, float], size: float
) -> Iterator[tuple[int, int]]:
    """Yield grid cells touched by the closed, axis-aligned search bbox."""
    for x in range(floor(bbox[0] / size), floor(bbox[2] / size) + 1):
        for y in range(floor(bbox[1] / size), floor(bbox[3] / size) + 1):
            yield x, y


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Test closed bbox overlap, including contact at a boundary."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _cross(a: PixelPoint, b: PixelPoint) -> float:
    """Return the 2D determinant ax * by - ay * bx used to intersect lines."""
    return a[0] * b[1] - a[1] * b[0]


def _intersection(
    first: TopologyGeometry,
    second: TopologyGeometry,
    endpoint_tolerance: float,
    max_extension: float,
    *,
    uses_context: bool,
) -> LineIntersection | None:
    """Bound the line intersection, classify its arms and preserve ID/metric order."""
    # 1. Intersect the supporting infinite lines; parallel lines have no unique hit.
    along = _intersection_distances(first, second)
    if along is None:
        return None
    # 2. e = max(0, -t, t - length) is the extension required by each finite line.
    extensions = tuple(
        max(0.0, -value, value - item.length)
        for value, item in zip(along, (first, second))
    )
    if any(value > max_extension + _PIXEL_EPSILON for value in extensions):
        return None
    # 3. Count arms longer than the endpoint tolerance: one = end, two = interior.
    # Zero arms means this line is too short around the hit to classify reliably.
    arm_counts = tuple(
        int(value > endpoint_tolerance) + int(item.length - value > endpoint_tolerance)
        for value, item in zip(along, (first, second))
    )
    if 0 in arm_counts:
        return None
    roles: tuple[Literal["end", "interior"], ...] = tuple(
        "interior" if count == 2 else "end" for count in arm_counts
    )
    # 4. An unmatched edge may explain an axis ending, not a crossing of its interior.
    if uses_context and roles[0] != "end":
        return None
    # 5. Recover the pixel position from the first line's signed distance.
    position = (
        first.line.start[0] + along[0] * first.unit[0],
        first.line.start[1] + along[0] * first.unit[1],
    )
    if not all(isfinite(value) for value in position):
        raise ValueError("Intersection position must be finite.")

    # 6. Sort IDs together with their roles/extensions to preserve source alignment.
    first_index, second_index = (
        (0, 1) if first.line.segment_id < second.line.segment_id else (1, 0)
    )
    line_ids = (first.line.segment_id, second.line.segment_id)
    topologies: tuple[Literal["L"], Literal["T"], Literal["X"]] = ("L", "T", "X")
    extensions = tuple(
        0.0 if value <= _PIXEL_EPSILON else value for value in extensions
    )
    return LineIntersection(
        position=position,
        line_ids=(line_ids[first_index], line_ids[second_index]),
        topology=topologies[roles.count("interior")],
        roles=(roles[first_index], roles[second_index]),
        extension_px=(extensions[first_index], extensions[second_index]),
        uses_unpaired_context=uses_context,
    )


def _intersection_distances(
    first: TopologyGeometry, second: TopologyGeometry
) -> tuple[float, float] | None:
    """Return signed distances along supporting lines, or no unique intersection."""
    denominator = _cross(first.unit, second.unit)
    if abs(denominator) <= _PARALLEL_EPSILON:
        return None  # Parallel/collinear lines have no unique L/T/X node.
    offset = (
        second.line.start[0] - first.line.start[0],
        second.line.start[1] - first.line.start[1],
    )
    # Solve p + t*u = q + s*v using determinants; unit vectors make t/s pixels.
    along = (
        _cross(offset, second.unit) / denominator,
        _cross(offset, first.unit) / denominator,
    )
    if not all(isfinite(value) for value in along):
        raise ValueError("Intersection arithmetic must be finite.")
    return along


def _deduplicate(
    hits: list[LineIntersection], radius: float
) -> tuple[JunctionCandidate, ...]:
    """Group hits around fixed anchors, preserving stronger evidence first."""
    # Prefer hits within fitted extents, then axis-only evidence; these are not scores.
    ordered = sorted(
        hits,
        key=lambda hit: (
            hit.hypothetical,
            hit.uses_unpaired_context,
            hit.position,
            hit.line_ids,
        ),
    )
    anchors: list[PixelPoint] = []
    clusters: list[list[LineIntersection]] = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    size = max(radius, _PIXEL_EPSILON)
    for hit in ordered:
        scaled = tuple(value / size for value in hit.position)
        if not all(isfinite(value) for value in scaled):
            raise ValueError("Coordinates exceed the deduplication grid range.")
        x, y = (floor(value) for value in scaled)
        # Anchors never move: attaching this hit cannot extend the cluster by a chain.
        anchor_index = _nearest_anchor(hit.position, anchors, grid, (x, y), radius)
        if anchor_index is not None:
            clusters[anchor_index].append(hit)
        else:
            grid[x, y].append(len(anchors))
            anchors.append(hit.position)
            clusters.append([hit])
    # Allocate local node IDs by anchor position, independent of input traversal.
    return tuple(
        JunctionCandidate(node_id, anchors[index], tuple(clusters[index]))
        for node_id, index in enumerate(
            sorted(range(len(anchors)), key=lambda i: anchors[i])
        )
    )


def _nearest_anchor(
    position: PixelPoint,
    anchors: list[PixelPoint],
    grid: dict[tuple[int, int], list[int]],
    cell: tuple[int, int],
    radius: float,
) -> int | None:
    """Choose the closest nearby anchor, using its insertion order for exact ties."""
    nearest: tuple[float, int] | None = None
    cell_x, cell_y = cell
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for index in grid.get((cell_x + offset_x, cell_y + offset_y), ()):
                anchor_x, anchor_y = anchors[index]
                distance = hypot(position[0] - anchor_x, position[1] - anchor_y)
                candidate = (distance, index)
                if distance <= radius and (nearest is None or candidate < nearest):
                    nearest = candidate
    return None if nearest is None else nearest[1]

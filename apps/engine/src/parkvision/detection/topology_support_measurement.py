"""Measure raw LSD coverage without changing or filtering topology candidates."""

from collections.abc import Mapping
from math import fsum, isfinite

from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .marking_topology import prepare_topology_geometry
from .segment_group import SegmentGroup
from .segment_merging import PixelPoint
from .topology_candidate import TopologyCandidates
from .topology_geometry import TopologyGeometry
from .topology_support import (
    EdgeSourceSupport,
    IntersectionSourceSupport,
    LineSourceSupport,
    SourceWindowSupport,
)

# Each interval contains start/end distances along a line and the original LSD ID.
type _SourceIntervals = tuple[tuple[float, float, int], ...]


def measure_topology_support(
    topology: TopologyCandidates,
    pairs: tuple[MarkingPair, ...],
    source_groups: Mapping[int, SegmentGroup],
    *,
    junction_radius_px: float = 3.0,
    arm_length_px: float = 15.0,
) -> tuple[IntersectionSourceSupport, ...]:
    """Project ORIGINAL group fragments onto each hit's axis/context direction.

    source_groups maps fitted EDGE IDs to their unchanged SegmentGroups, never
    LSD IDs or centerline IDs. Require complete provenance, including isolated
    lines. Missing/invalid sources raise, rather than appearing as zero evidence.
    Each edge has three fixed windows: [-radius, radius] around the exact hit,
    [-length, -radius] before it and [radius, length] after it. Neither source
    intervals nor windows are clipped to the centerline's shared extent: evidence
    beyond that extent may explain an extrapolated hypothesis. Record both sides
    even for an end role; an absent arm naturally has low/zero source coverage.

    This measures longitudinal fragment support, NOT paint, contrast, transverse
    proximity, a probability, or a new L/T/X classification. Even perfect support
    can come from a curb/pole/shadow. No threshold removes lines or candidates.
    """
    # 1. Validate windows and resolve every topology line to its original edge groups.
    if not (
        isfinite(junction_radius_px)
        and isfinite(arm_length_px)
        and isfinite(2 * junction_radius_px)
        and 0 < junction_radius_px < arm_length_px
    ):
        raise ValueError("Require finite 0 < junction_radius_px < arm_length_px.")

    edges_by_line = _resolve_source_edges(topology, pairs)
    edge_ids = {edge.segment_id for edges in edges_by_line.values() for edge in edges}
    if set(source_groups) != edge_ids:
        raise ValueError("Source groups must match all paired and unpaired edge IDs.")

    topology_line_ids = {line.segment_id for line in topology.lines}
    _validate_source_groups(source_groups, topology_line_ids | edge_ids)

    # 2. Reuse topology geometry and project each source group once, before any hits.
    geometry_by_line = {
        line.segment_id: prepare_topology_geometry(line) for line in topology.lines
    }
    projected_sources = _project_source_groups(
        geometry_by_line, edges_by_line, source_groups
    )

    # 3. Measure exact pairwise intersections, rather than the shared node anchor.
    unpaired_line_ids = set(topology.unpaired_line_ids)
    measurements = []
    for junction in topology.junctions:
        for intersection_index, intersection in enumerate(junction.intersections):
            line_measurements = []
            for line_id in intersection.line_ids:
                line_measurements.append(
                    _measure_line_support(
                        geometry_by_line[line_id],
                        edges_by_line[line_id],
                        projected_sources,
                        intersection.position,
                        uses_unpaired_context=line_id in unpaired_line_ids,
                        junction_radius_px=junction_radius_px,
                        arm_length_px=arm_length_px,
                    )
                )

            # A pairwise intersection always has exactly two participating lines.
            first_line, second_line = line_measurements
            measurements.append(
                IntersectionSourceSupport(
                    candidate_id=junction.candidate_id,
                    intersection_index=intersection_index,
                    lines=(first_line, second_line),
                )
            )
    return tuple(measurements)


def _resolve_source_edges(
    topology: TopologyCandidates, pairs: tuple[MarkingPair, ...]
) -> dict[int, tuple[LineSegment, ...]]:
    """Map axes to their two fitted edges and context lines to their single edge."""
    # 1. Check that axes in the topology are exactly those supplied by the pairs.
    pairs_by_axis_id = {pair.centerline.segment_id: pair for pair in pairs}
    lines_by_id = {line.segment_id: line for line in topology.lines}
    unpaired_line_ids = set(topology.unpaired_line_ids)
    axis_ids = lines_by_id.keys() - unpaired_line_ids
    if (
        len(pairs_by_axis_id) != len(pairs)
        or len(lines_by_id) != len(topology.lines)
        or set(pairs_by_axis_id) != axis_ids
        or not unpaired_line_ids <= lines_by_id.keys()
        or any(
            lines_by_id[line_id] != pair.centerline
            for line_id, pair in pairs_by_axis_id.items()
        )
    ):
        raise ValueError("Topology lines and marking pair identities must agree.")

    # 2. Retain both paired edges independently; an unpaired line is its own edge.
    edges_by_line: dict[int, tuple[LineSegment, ...]] = {}
    for line_id, line in lines_by_id.items():
        pair = pairs_by_axis_id.get(line_id)
        edges_by_line[line_id] = pair.edges if pair is not None else (line,)

    edge_ids = [edge.segment_id for edges in edges_by_line.values() for edge in edges]
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("Fitted edge IDs must identify distinct source groups.")
    return edges_by_line


def _project_source_groups(
    geometry_by_line: Mapping[int, TopologyGeometry],
    edges_by_line: Mapping[int, tuple[LineSegment, ...]],
    source_groups: Mapping[int, SegmentGroup],
) -> dict[tuple[int, int], _SourceIntervals]:
    """Cache source intervals by (topology line ID, fitted edge ID)."""
    projected_sources = {}
    for line_id, line_edges in edges_by_line.items():
        geometry = geometry_by_line[line_id]
        for edge in line_edges:
            source_intervals = []
            for segment in source_groups[edge.segment_id].segments:
                start_distance = _project_point(segment.start, geometry)
                end_distance = _project_point(segment.end, geometry)
                if not (isfinite(start_distance) and isfinite(end_distance)):
                    raise ValueError("Projected source geometry must be finite.")
                source_intervals.append(
                    (
                        min(start_distance, end_distance),
                        max(start_distance, end_distance),
                        segment.segment_id,
                    )
                )
            sorted_intervals = tuple(sorted(source_intervals))
            projected_sources[line_id, edge.segment_id] = sorted_intervals
    return projected_sources


def _project_point(point: PixelPoint, geometry: TopologyGeometry) -> float:
    """Return signed distance along the line, measured from its stored start."""
    offset_x = point[0] - geometry.line.start[0]
    offset_y = point[1] - geometry.line.start[1]
    return fsum((offset_x * geometry.unit[0], offset_y * geometry.unit[1]))


def _measure_line_support(
    geometry: TopologyGeometry,
    edges: tuple[LineSegment, ...],
    projected_sources: Mapping[tuple[int, int], _SourceIntervals],
    intersection_position: PixelPoint,
    *,
    uses_unpaired_context: bool,
    junction_radius_px: float,
    arm_length_px: float,
) -> LineSourceSupport:
    """Measure the junction and signed arms separately for each source edge."""
    intersection_distance = _project_point(intersection_position, geometry)
    edge_measurements = []
    for edge in edges:
        # 1. Move the distance origin to this exact intersection, keeping source gaps.
        source_intervals = projected_sources[geometry.line.segment_id, edge.segment_id]
        relative_intervals = tuple(
            (start - intersection_distance, end - intersection_distance, source_id)
            for start, end, source_id in source_intervals
        )
        nearest_fragment_gap = min(
            max(0.0, start, -end) for start, end, _ in relative_intervals
        )

        # 2. Use full windows on both sides, including outside the fitted axis extent.
        junction_support = _measure_window_support(
            relative_intervals, -junction_radius_px, junction_radius_px
        )
        before_support = _measure_window_support(
            relative_intervals, -arm_length_px, -junction_radius_px
        )
        after_support = _measure_window_support(
            relative_intervals, junction_radius_px, arm_length_px
        )
        edge_measurements.append(
            EdgeSourceSupport(
                fitted_edge_id=edge.segment_id,
                nearest_fragment_gap_px=nearest_fragment_gap,
                junction=junction_support,
                before=before_support,
                after=after_support,
            )
        )

    return LineSourceSupport(
        line_id=geometry.line.segment_id,
        kind="unpaired_context" if uses_unpaired_context else "axis",
        edges=tuple(edge_measurements),
    )


def _validate_source_groups(
    groups: Mapping[int, SegmentGroup], derived_ids: set[int]
) -> None:
    """Reject missing fragments, ambiguous source IDs and invalid raw geometry."""
    seen_source_ids = set()
    for group in groups.values():
        if not group.segments:
            raise ValueError("Source groups must contain original LSD fragments.")
        for segment in group.segments:
            # Source identity must stay distinct from every fitted/axis identity.
            if (
                segment.segment_id in seen_source_ids
                or segment.segment_id in derived_ids
            ):
                raise ValueError(
                    "Raw source IDs must be unique and separate from fitted IDs."
                )
            seen_source_ids.add(segment.segment_id)
            if not (
                all(isfinite(value) for value in (*segment.start, *segment.end))
                and isfinite(segment.length)
                and segment.length > 0
            ):
                raise ValueError(
                    "Raw source fragments must have finite, non-zero geometry."
                )


def _measure_window_support(
    intervals: _SourceIntervals, window_start: float, window_end: float
) -> SourceWindowSupport:
    """Measure interval union, without double-counting overlaps or filling gaps."""
    covered_lengths = []
    contributing_source_ids = []
    covered_until = window_start

    # Intervals arrive sorted by start. Track the union's right edge so overlapping
    # fragments contribute their shared length only once.
    for source_start, source_end, source_id in intervals:
        overlap_start = max(window_start, source_start)
        overlap_end = min(window_end, source_end)
        if overlap_start >= overlap_end:
            continue

        contributing_source_ids.append(source_id)
        uncovered_start = max(covered_until, overlap_start)
        covered_lengths.append(max(0.0, overlap_end - uncovered_start))
        covered_until = max(covered_until, overlap_end)

    return SourceWindowSupport(
        window_px=(window_start, window_end),
        covered_length_px=min(window_end - window_start, fsum(covered_lengths)),
        source_segment_ids=tuple(sorted(contributing_source_ids)),
    )

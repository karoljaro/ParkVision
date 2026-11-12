"""Join local fragments of a marking while retaining the original LSD evidence."""

from collections.abc import Mapping

from .line_fitting import fit_group_line
from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .marking_pairing import pair_marking_edges
from .segment_group import SegmentGroup
from .segment_merging import group_segments


def pair_marking_fragments(
    fitted_lines: tuple[LineSegment, ...],
    source_groups: Mapping[int, SegmentGroup],
    scale: float,
) -> tuple[tuple[LineSegment, ...], dict[int, SegmentGroup], tuple[MarkingPair, ...]]:
    """Re-fit both sides of nearby collinear paired fragments where possible.

    The combined fitted extents are hypotheses. Callers still measure coverage
    against the unmodified LSD fragments, so this step cannot fill a gap.
    """
    pairs = _pair_edges(fitted_lines, scale)
    by_axis = {pair.centerline.segment_id: pair for pair in pairs}
    tracks = group_segments(
        tuple(pair.centerline for pair in pairs),
        angle_tolerance=2.5,
        distance_tolerance=min(4.0, 3.2 * scale),
        gap_tolerance=min(51.0, 42.0 * scale),
    )
    lines = {line.segment_id: line for line in fitted_lines}
    sources = dict(source_groups)
    next_id = max((*lines, *by_axis), default=-1) + 1
    joined_pairs = []
    for track in tracks:
        members = tuple(by_axis[line.segment_id] for line in track.segments)
        if len(members) == 1:
            joined_pairs.extend(members)
            continue

        guide = fit_group_line(track, segment_id=next_id)
        if guide.length <= 1.15 * max(line.length for line in track.segments):
            joined_pairs.extend(members)
            continue
        nx = guide.start[1] - guide.end[1]
        ny = guide.end[0] - guide.start[0]

        def across(edge: LineSegment) -> float:
            return nx * (edge.start[0] + edge.end[0]) + ny * (
                edge.start[1] + edge.end[1]
            )

        ordered_edges = tuple(sorted(pair.edges, key=across) for pair in members)
        sides = tuple(
            SegmentGroup(
                tuple(
                    fragment
                    for edge in side
                    for fragment in source_groups[edge.segment_id].segments
                )
            )
            for side in zip(*ordered_edges)
        )
        edges = tuple(
            fit_group_line(side, segment_id=next_id + offset)
            for offset, side in enumerate(sides)
        )
        merged = _pair_edges(edges, scale)
        if not merged:
            joined_pairs.extend(members)
            continue

        for pair in members:
            for edge in pair.edges:
                del lines[edge.segment_id]
                del sources[edge.segment_id]
        for edge, side in zip(edges, sides):
            lines[edge.segment_id] = edge
            sources[edge.segment_id] = side
        joined_pairs.extend(merged)
        next_id += 3  # Two fitted edges and the paired axis.

    return tuple(lines.values()), sources, tuple(joined_pairs)


def _pair_edges(
    edges: tuple[LineSegment, ...], scale: float
) -> tuple[MarkingPair, ...]:
    return pair_marking_edges(
        edges,
        min_width_px=max(1.25, 1.5 * scale),
        max_width_px=15 * scale,
        angle_tolerance=4.0,
        min_overlap_px=24 * scale,
        min_overlap_ratio=0.4,
        max_width_ratio=2.0,
    )

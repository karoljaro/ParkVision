"""Join local fragments of a marking while retaining the original LSD evidence."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from .line_fitting import fit_group_line
from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .marking_pairing import pair_marking_edges
from .segment_group import SegmentGroup
from .segment_merging import (
    angle_difference,
    group_segments,
    point_line_distance,
    projected_segment_gap,
)

_PROFILE_STEP_PX = 3.0


@dataclass(frozen=True, slots=True)
class _PaintTrace:
    axis: LineSegment
    source: LineSegment


@dataclass(slots=True)
class _MarkingTrack:
    axis: LineSegment
    traces: list[_PaintTrace]


def recover_marking_fragments(
    image: NDArray[np.uint8], segments: tuple[LineSegment, ...], scale: float
) -> tuple[tuple[LineSegment, ...], dict[int, SegmentGroup], tuple[MarkingPair, ...]]:
    """Use local paint profiles to group both sides before geometric pairing.

    LSD gives each directed edge its bright side. Cross-sections confirm a narrow
    bright stripe with a second transition. Their centers guide grouping, while
    fitted edges and all coverage measurements still use original LSD fragments.
    Unclaimed fragments remain available as weaker transverse context.
    """
    eligible = tuple(segment for segment in segments if segment.length >= 9 * scale)
    if not eligible:
        return (), {}, ()
    next_id = max(segment.segment_id for segment in segments) + 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    max_width = max(8.0, min(18.0, 14 * scale))
    traces = []
    for source in eligible:
        found = _trace_paint(gray, source, max_width, next_id)
        traces.extend(found)
        next_id += len(found)

    tracks = _join_traces(traces, scale, next_id)
    # The same ID is only a temporary fit ID inside _join_traces.
    lines = []
    sources = {}
    pairs = []
    claimed = set()
    for track in sorted(tracks, key=lambda track: -track.axis.length):
        available = [
            trace for trace in track.traces if trace.source.segment_id not in claimed
        ]
        coverage = defaultdict(float)
        for trace in available:
            coverage[trace.source.segment_id] += trace.axis.length + _PROFILE_STEP_PX
        supported = [
            trace
            for trace in available
            if coverage[trace.source.segment_id] >= 0.45 * trace.source.length
        ]
        if not supported:
            continue
        axis = fit_group_line(
            SegmentGroup(tuple(trace.axis for trace in supported)), segment_id=next_id
        )
        if axis.length < 24 * scale:
            continue
        sides = _track_sides(axis, supported)
        if not all(side.segments for side in sides):
            continue
        edges = tuple(
            fit_group_line(side, segment_id=next_id + i) for i, side in enumerate(sides)
        )
        matched = pair_marking_edges(
            edges,
            min_width_px=0.9,
            max_width_px=16 * scale,
            angle_tolerance=4.0,
            min_overlap_px=20 * scale,
            min_overlap_ratio=0.30,
            max_width_ratio=4.0,
        )
        if not matched:
            continue
        pair = matched[0]
        # A common stripe can continue along one edge where the opposite LSD edge
        # is intermittent. The guide extent only reaches measured cross-sections.
        pairs.append(
            replace(
                pair, centerline=replace(axis, segment_id=pair.centerline.segment_id)
            )
        )
        for edge, side in zip(edges, sides):
            lines.append(edge)
            sources[edge.segment_id] = side
            claimed.update(source.segment_id for source in side.segments)
        next_id = pair.centerline.segment_id + 1

    remaining = tuple(edge for edge in eligible if edge.segment_id not in claimed)
    for group in group_segments(remaining, 3.0, 2 * scale, 28 * scale):
        lines.append(fit_group_line(group, segment_id=next_id))
        sources[next_id] = group
        next_id += 1
    return tuple(lines), sources, tuple(pairs)


def _trace_paint(
    gray: NDArray[np.float32], source: LineSegment, max_width: float, first_id: int
) -> list[_PaintTrace]:
    # Sample from the dark side through the edge and across the possible stripe.
    start = np.array(source.start)
    direction = (np.array(source.end) - start) / source.length
    inward = np.array([direction[1], -direction[0]])
    along = np.linspace(
        0.8,
        source.length - 0.8,
        max(3, round(source.length / _PROFILE_STEP_PX) + 1),
    )
    offsets = np.arange(-3.0, max_width + 3.0, 0.5)
    positions = start + along[:, None] * direction
    grid = (positions[:, None, :] + offsets[None, :, None] * inward).astype(np.float32)
    profiles = cv2.remap(
        gray,
        grid[:, :, 0],
        grid[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    centers = []
    traces = []
    for position, profile in zip(positions, profiles):
        offset = _stripe_center(profile, max_width)
        if offset is not None:
            center = position + offset * inward
            centers.append((float(center[0]), float(center[1])))
            continue
        _append_trace(traces, centers, source, first_id)
        centers.clear()
    _append_trace(traces, centers, source, first_id)
    return traces


def _stripe_center(profile: NDArray[np.float32], max_width: float) -> float | None:
    # Background samples are -3 to -1.5 px. The peak is sought at +0.5 to +3 px.
    background = float(np.median(profile[:4]))
    peak = float(np.max(profile[7:13]))
    contrast = peak - background
    if contrast < 10:
        return None
    bright = profile >= (peak + background) / 2
    # Require a contiguous bright run through +1 px, bounded by two dark sides.
    if not bright[8]:
        return None
    left = np.flatnonzero(~bright[:8])
    right = np.flatnonzero(~bright[9:])
    if not left.size or not right.size:
        return None
    stop = int(right[0]) + 9
    near = -3 + 0.5 * (int(left[-1]) + 0.5)
    far = -3 + 0.5 * (stop - 0.5)
    if not (abs(near) <= 1.6 and 1.5 <= far - near <= max_width):
        return None
    if stop + 2 >= len(profile):
        return None
    outside = profile[stop : stop + 3]
    if contrast < 31:
        # Faint paint must form a clean, two-sided stripe in its local lighting.
        # A robust background noise estimate avoids promoting texture or a single
        # illumination step merely because its absolute contrast is small.
        if stop + 4 >= len(profile):
            return None
        # The first samples after the half-height crossing include the falling
        # edge itself. Estimate surface noise one pixel farther outside paint.
        background_tail = profile[stop + 2 : stop + 5]
        noise = max(
            1.0,
            float(np.median(np.abs(np.diff(profile[:4])))),
            float(np.median(np.abs(np.diff(background_tail)))),
        )
        far_background = float(np.median(background_tail))
        if contrast < 6 * noise or abs(far_background - background) > max(
            3.0, 0.25 * contrast
        ):
            return None
    required_fall = 25.0 if contrast >= 31 else 0.75 * contrast
    if float(np.min(outside)) > peak - required_fall:
        return None
    return (near + far) / 2


def _append_trace(
    traces: list[_PaintTrace],
    centers: list[tuple[float, float]],
    source: LineSegment,
    first_id: int,
) -> None:
    if len(centers) >= 3:
        axis = LineSegment(first_id + len(traces), centers[0], centers[-1])
        traces.append(_PaintTrace(axis, source))


def _join_traces(
    traces: list[_PaintTrace], scale: float, fit_id: int
) -> list[_MarkingTrack]:
    tracks: list[_MarkingTrack] = []
    gap = min(36.0, 31 * scale)
    residual = min(3.5, max(1.4, 2.8 * scale))
    for trace in sorted(traces, key=lambda item: -item.axis.length):
        axis = trace.axis
        for track in tracks:
            guide = track.axis
            if (
                angle_difference(guide, axis) > 4
                or projected_segment_gap(guide, axis) > gap
                or min(point_line_distance(p, guide) for p in (axis.start, axis.end))
                > 5 * scale
            ):
                continue
            members = tuple(member.axis for member in track.traces) + (axis,)
            fitted = fit_group_line(SegmentGroup(members), segment_id=fit_id)
            if any(
                point_line_distance(point, fitted) > residual
                for member in members
                for point in (member.start, member.end)
            ):
                continue
            track.axis = fitted
            track.traces.append(trace)
            break
        else:
            tracks.append(_MarkingTrack(axis, [trace]))
    return tracks


def _track_sides(
    axis: LineSegment, traces: list[_PaintTrace]
) -> tuple[SegmentGroup, SegmentGroup]:
    dx, dy = axis.end[0] - axis.start[0], axis.end[1] - axis.start[1]
    forward = []
    backward = []
    for trace in traces:
        source = trace.source
        alignment = (source.end[0] - source.start[0]) * dx + (
            source.end[1] - source.start[1]
        ) * dy
        (forward if alignment > 0 else backward).append(source)
    return SegmentGroup(tuple(dict.fromkeys(backward))), SegmentGroup(
        tuple(dict.fromkeys(forward))
    )


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

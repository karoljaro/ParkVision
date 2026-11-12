"""Assemble parking rows from independently supported marking geometry."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import cos, hypot, radians

import cv2
import numpy as np
from numpy.typing import NDArray

from ..domain.geometry.point import Point
from ..domain.geometry.polygon import Polygon
from ..domain.types import CameraId
from .initialization import BoundaryEvidence, SlotCandidate
from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .segment_group import SegmentGroup
from .segment_merging import PixelPoint
from .slot_proposal import SlotProposal
from .topology_candidate import TopologyCandidates
from .topology_support import IntersectionSourceSupport, LineSourceSupport

_MIN_LOCAL_ALIGNMENT = cos(radians(24))
_MAX_TRANSVERSE_ALIGNMENT = cos(radians(23))
_MIN_BACK_ALIGNMENT = cos(radians(55))
_MAX_NEIGHBOR_WIDTH_RATIO = 2.0
_EPSILON = 1e-7


@dataclass(frozen=True, slots=True)
class _Anchor:
    rail_id: int
    separator_id: int
    node_id: int
    position: PixelPoint
    tip: PixelPoint
    direction: PixelPoint
    position_on_rail: float

    @property
    def depth(self) -> float:
        return _distance(self.position, self.tip)


@dataclass(frozen=True, slots=True)
class _MarkingEvidence:
    line: LineSegment
    direction: PixelPoint
    intervals: tuple[tuple[float, float], ...]
    width_px: float = 0.0


@dataclass(frozen=True, slots=True)
class _Closure:
    first: PixelPoint
    second: PixelPoint
    rail_id: int | None = None
    node_ids: tuple[int, ...] = ()
    context_ids: tuple[int, ...] = ()


def assemble_slots(
    camera_id: CameraId,
    width: int,
    height: int,
    pairs: tuple[MarkingPair, ...],
    topology: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    source_groups: Mapping[int, SegmentGroup],
) -> tuple[SlotCandidate, ...]:
    """Use shared transverse anchors and adjacent separators to bound polygons.

    A third separator corroborates the row. A shared rear marking is preferred.
    Otherwise aligned tips bound closure, with short extrapolation allowed for
    an internal, bracketed tip. Missing source spans are recorded separately.
    """
    diagonal = hypot(width, height)
    min_depth = max(18.0, 0.021 * diagonal)
    min_width = max(10.0, 0.008 * diagonal)
    noise_px = min(3.0, max(1.25, diagonal * 0.0018))
    evidence = _marking_evidence(topology, pairs, source_groups, noise_px)
    rows, anchor_index = _anchors(topology, support, pairs, evidence, min_depth)
    proposed = []
    for key in sorted(rows):
        row = sorted(rows[key], key=lambda anchor: anchor.position_on_rail)
        if len({anchor.separator_id for anchor in row}) < 3:
            continue
        gaps = [
            (
                _distance(left.position, right.position)
                if _compatible(left, right, min_width)
                else None
            )
            for left, right in zip(row, row[1:])
        ]
        for index, gap in enumerate(gaps):
            if gap is None:
                continue
            first, second = row[index : index + 2]
            context = _neighbor_indexes(gaps, index, gap)
            if not context:
                proposed.append(
                    _unresolved(first, second, "No third compatible separator")
                )
                continue
            closure = _observed_closure(first, second, anchor_index, evidence)
            if closure is None:
                closure, reason = _estimated_closure(
                    row, index, context, gap, evidence, width, height, noise_px
                )
                if closure is None:
                    proposed.append(_unresolved(first, second, reason))
                    continue
            proposed.append(
                _build_candidate(
                    camera_id,
                    width,
                    height,
                    first,
                    second,
                    closure,
                    evidence,
                    gap,
                    min_width,
                    noise_px,
                )
            )

    if not proposed:
        return (
            SlotCandidate(
                "unresolved", None, (), (), "No row with adjacent supported separators"
            ),
        )
    return _select_nonoverlapping(proposed, width, height)


def _marking_evidence(
    topology: TopologyCandidates,
    pairs: tuple[MarkingPair, ...],
    source_groups: Mapping[int, SegmentGroup],
    noise_px: float,
) -> dict[int, _MarkingEvidence]:
    by_axis = {pair.centerline.segment_id: pair for pair in pairs}
    markings = {}
    for line in topology.lines:
        direction = _direction(line.start, line.end)
        pair = by_axis.get(line.segment_id)
        edge_ids = (
            (line.segment_id,)
            if pair is None
            else tuple(edge.segment_id for edge in pair.edges)
        )
        intervals = []
        for edge_id in edge_ids:
            for fragment in source_groups[edge_id].segments:
                start, end = sorted(
                    _dot(_subtract(point, line.start), direction)
                    for point in (fragment.start, fragment.end)
                )
                intervals.append((max(0.0, start), min(line.length, end)))
        joined: list[tuple[float, float]] = []
        for start, end in sorted(intervals):
            if end <= start:
                continue
            if joined and start <= joined[-1][1] + noise_px:
                joined[-1] = (joined[-1][0], max(end, joined[-1][1]))
            else:
                joined.append((start, end))
        markings[line.segment_id] = _MarkingEvidence(
            line, direction, tuple(joined), 0.0 if pair is None else min(pair.widths_px)
        )
    return markings


def _anchors(
    topology: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    pairs: tuple[MarkingPair, ...],
    evidence: Mapping[int, _MarkingEvidence],
    min_depth: float,
) -> tuple[dict[tuple[int, int], list[_Anchor]], dict[int, list[_Anchor]]]:
    paired = {pair.centerline.segment_id for pair in pairs}
    endpoints = defaultdict(list)
    for endpoint in topology.endpoints:
        endpoints[endpoint.line_id].append(endpoint.position)
    measurements = {
        (item.candidate_id, item.intersection_index): item for item in support
    }
    rows = defaultdict(list)
    by_separator = defaultdict(list)
    for node in topology.junctions:
        for index, hit in enumerate(node.intersections):
            measurement = measurements.get((node.candidate_id, index))
            if measurement is None or not all(
                _supported(line) for line in measurement.lines
            ):
                continue
            for rail_id, separator_id in (hit.line_ids, hit.line_ids[::-1]):
                if separator_id not in paired:
                    continue
                rail = evidence[rail_id]
                separator = evidence[separator_id]
                if abs(_dot(rail.direction, separator.direction)) > (
                    _MAX_TRANSVERSE_ALIGNMENT
                ):
                    continue
                for tip in endpoints[separator_id]:
                    if _distance(hit.position, tip) < min_depth:
                        continue
                    direction = _direction(hit.position, tip)
                    side = 1 if _cross(rail.direction, direction) > 0 else -1
                    anchor = _Anchor(
                        rail_id,
                        separator_id,
                        node.candidate_id,
                        hit.position,
                        tip,
                        direction,
                        _dot(_subtract(hit.position, rail.line.start), rail.direction),
                    )
                    rows[rail_id, side].append(anchor)
                    by_separator[separator_id].append(anchor)
    return rows, by_separator


def _supported(measurement: LineSourceSupport) -> bool:
    return any(
        edge.nearest_fragment_gap_px <= edge.junction.window_px[1] + 2.0
        and max(edge.before.coverage_ratio, edge.after.coverage_ratio) >= 0.34
        for edge in measurement.edges
    )


def _compatible(first: _Anchor, second: _Anchor, min_width: float) -> bool:
    gap = _distance(first.position, second.position)
    return (
        first.separator_id != second.separator_id
        and min_width <= gap <= 2.4 * min(first.depth, second.depth)
        and _dot(first.direction, second.direction) >= _MIN_LOCAL_ALIGNMENT
    )


def _neighbor_indexes(
    gaps: list[float | None], index: int, gap: float
) -> tuple[int, ...]:
    neighbors = []
    for adjacent, separator in ((index - 1, index - 1), (index + 1, index + 2)):
        if 0 <= adjacent < len(gaps) and (other := gaps[adjacent]) is not None:
            if max(other, gap) <= _MAX_NEIGHBOR_WIDTH_RATIO * min(other, gap):
                neighbors.append(separator)
    return tuple(neighbors)


def _observed_closure(
    first: _Anchor,
    second: _Anchor,
    anchor_index: Mapping[int, list[_Anchor]],
    evidence: Mapping[int, _MarkingEvidence],
) -> _Closure | None:
    front_direction = evidence[first.rail_id].direction
    best = None
    best_depth = float("inf")
    for left in anchor_index[first.separator_id]:
        if (
            left.rail_id == first.rail_id
            or abs(_dot(front_direction, evidence[left.rail_id].direction))
            < _MIN_BACK_ALIGNMENT
        ):
            continue
        depth_first = _dot(_subtract(left.position, first.position), first.direction)
        if not 0.44 * first.depth <= depth_first <= first.depth + 5:
            continue
        for right in anchor_index[second.separator_id]:
            if left.rail_id != right.rail_id:
                continue
            depth_second = _dot(
                _subtract(right.position, second.position), second.direction
            )
            if not 0.44 * second.depth <= depth_second <= second.depth + 5:
                continue
            if not 0.55 <= depth_first / depth_second <= 1.82:
                continue
            if (depth := depth_first + depth_second) < best_depth:
                best_depth = depth
                best = _Closure(
                    left.position,
                    right.position,
                    left.rail_id,
                    (left.node_id, right.node_id),
                )
    return best


def _estimated_closure(
    row: list[_Anchor],
    index: int,
    context: tuple[int, ...],
    gap: float,
    evidence: Mapping[int, _MarkingEvidence],
    width: int,
    height: int,
    noise_px: float,
) -> tuple[_Closure | None, str]:
    first, second = row[index : index + 2]
    first_tip = _recover_tip(row, index, gap, width, height) or first.tip
    second_tip = _recover_tip(row, index + 1, gap, width, height) or second.tip
    if not all(_interior(point, width, height) for point in (first_tip, second_tip)):
        return None, "Separator tip is clipped or too close to the image boundary"
    if any(_has_continuation(anchor, evidence) for anchor in (first, second)):
        return None, "A nearby continuation makes the apparent separator tip ambiguous"
    back_gap = _distance(first_tip, second_tip)
    if not 0.46 * gap <= back_gap <= 2.2 * gap:
        return None, "Inconsistent spacing between front and rear boundaries"
    if min(first.depth, second.depth) < 1.05 * max(gap, back_gap):
        return None, "Short transverse pattern lacks independent rear support"

    direction = _direction(first_tip, second_tip)
    if _dot(evidence[first.rail_id].direction, direction) < _MIN_BACK_ALIGNMENT:
        return None, "Separator tips do not align with the row"
    tolerance = min(10.0, max(2 * noise_px, 0.12 * back_gap))
    for neighbor in _closure_neighbors(row, index, context, gap):
        other = row[neighbor]
        other_tip = _recover_tip(row, neighbor, gap, width, height) or other.tip
        if (
            _interior(other_tip, width, height)
            and abs(_cross(_subtract(other_tip, first_tip), direction)) <= tolerance
            and not _has_continuation(other, evidence)
        ):
            return (
                _Closure(
                    first_tip,
                    second_tip,
                    node_ids=(other.node_id,),
                    context_ids=(other.separator_id,),
                ),
                "",
            )
    return None, "Rear boundary lacks a common rail or a corroborating separator tip"


def _closure_neighbors(
    row: list[_Anchor], index: int, context: tuple[int, ...], gap: float
) -> tuple[int, ...]:
    """A short neighbor can still anchor spacing for the next visible tip."""
    result = list(context)
    depth = min(row[index].depth, row[index + 1].depth)
    for neighbor in context:
        step = -1 if neighbor < index else 1
        beyond = neighbor + step
        if row[neighbor].depth >= 0.65 * depth or not 0 <= beyond < len(row):
            continue
        if _compatible(row[neighbor], row[beyond], 0.5 * gap):
            spacing = _distance(row[neighbor].position, row[beyond].position)
            if spacing <= _MAX_NEIGHBOR_WIDTH_RATIO * gap:
                result.append(beyond)
    return tuple(result)


def _has_continuation(
    anchor: _Anchor, evidence: Mapping[int, _MarkingEvidence]
) -> bool:
    # Test fitted hypotheses against original fragment coverage beyond the tip.
    search_gap = min(48.0, max(10.0, 0.32 * anchor.depth))
    lateral = min(8.0, max(3.0, 1.5 * evidence[anchor.separator_id].width_px))
    for line_id, other in evidence.items():
        if line_id in (anchor.separator_id, anchor.rail_id):
            continue
        if abs(_dot(anchor.direction, other.direction)) < cos(radians(5)):
            continue
        for lo, hi in other.intervals:
            points = tuple(
                _advance(other.line.start, other.direction, t) for t in (lo, hi)
            )
            offsets = [_subtract(point, anchor.tip) for point in points]
            near, far = sorted(_dot(offset, anchor.direction) for offset in offsets)
            if (
                far >= 9.0
                and near <= search_gap
                and max(abs(_cross(offset, anchor.direction)) for offset in offsets)
                <= lateral
            ):
                return True
    return False


def _recover_tip(
    row: list[_Anchor], index: int, width_px: float, width: int, height: int
) -> PixelPoint | None:
    if index == 0 or index == len(row) - 1:
        return None
    before, center, after = row[index - 1 : index + 2]
    if not (
        _compatible(before, center, 0.45 * width_px)
        and _compatible(center, after, 0.45 * width_px)
        and _interior(before.tip, width, height)
        and _interior(after.tip, width, height)
        and center.depth < 0.96 * min(before.depth, after.depth)
    ):
        return None
    expected_back = _subtract(after.tip, before.tip)
    denominator = _cross(center.direction, expected_back)
    if abs(denominator) < 0.35 * hypot(*expected_back):
        return None
    offset = _subtract(before.tip, center.position)
    reach = _cross(offset, expected_back) / denominator
    fraction = _cross(offset, center.direction) / denominator
    extension = reach - center.depth
    budget = min(28.0, 0.18 * min(before.depth, after.depth), 0.44 * width_px)
    if 0.13 <= fraction <= 0.87 and 0 < extension <= budget:
        return _advance(center.position, center.direction, reach)
    return None


def _build_candidate(
    camera_id: CameraId,
    width: int,
    height: int,
    first: _Anchor,
    second: _Anchor,
    closure: _Closure,
    evidence: Mapping[int, _MarkingEvidence],
    gap: float,
    min_width: float,
    noise_px: float,
) -> SlotCandidate:
    points = first.position, second.position, closure.second, closure.first
    back_gap = _distance(closure.first, closure.second)
    if not (
        max(min_width * 0.65, 0.38 * gap) <= back_gap <= 2.55 * gap
        and all(0 <= x <= width - 1 and 0 <= y <= height - 1 for x, y in points)
        and _convex(points)
    ):
        return _unresolved(first, second, "Inconsistent or off-frame boundaries")
    if _has_inner_separator(points, first, second, evidence, gap):
        return _unresolved(first, second, "Visible inner separator divides this area")
    try:
        polygon = Polygon(
            tuple(Point(x / (width - 1), y / (height - 1)) for x, y in points)
        )
    except ValueError:
        return _unresolved(first, second, "Boundary intersections are not a valid ROI")

    line_ids = first.rail_id, second.separator_id, closure.rail_id, first.separator_id
    boundaries = []
    for index, (start, end, line_id) in enumerate(
        zip(points, points[1:] + points[:1], line_ids)
    ):
        if line_id is None:
            boundaries.append(
                BoundaryEvidence(
                    start,
                    end,
                    "estimated",
                    (first.separator_id, second.separator_id, *closure.context_ids),
                )
            )
            continue
        crossing = (line_ids[index - 1], line_ids[(index + 1) % len(line_ids)])
        corner_widths = tuple(
            0.0 if line is None else min(8.0, evidence[line].width_px)
            for line in crossing
        )
        spans = _boundary_spans(start, end, evidence[line_id], noise_px, corner_widths)
        length = _distance(start, end)
        missing = sum(
            _distance(span.start, span.end)
            for span in spans
            if span.kind == "estimated"
        )
        if missing > min(28.0, 0.26 * length, 0.45 * gap):
            return _unresolved(
                first, second, "Insufficient source fragments on a boundary"
            )
        boundaries.extend(spans)

    estimated = any(span.kind == "estimated" for span in boundaries)
    if closure.rail_id is None:
        reason = (
            "Rear boundary joins aligned separator tips; short gaps may be estimated"
        )
    elif estimated:
        reason = "Shared transverse rails with bounded gaps in source fragments"
    else:
        reason = "Adjacent separators and two transverse rails have source support"
    return SlotCandidate(
        "estimated" if estimated else "observed",
        SlotProposal(camera_id, polygon),
        tuple(boundaries),
        tuple(sorted({first.node_id, second.node_id, *closure.node_ids})),
        reason,
    )


def _has_inner_separator(
    points: tuple[PixelPoint, ...],
    first: _Anchor,
    second: _Anchor,
    evidence: Mapping[int, _MarkingEvidence],
    gap: float,
) -> bool:
    contour = np.array(points, dtype=np.float32)
    margin = min(8.0, max(3.0, 0.15 * gap))
    min_support = max(12.0, 0.18 * min(first.depth, second.depth))
    for line_id, other in evidence.items():
        if (
            line_id in (first.separator_id, second.separator_id)
            or other.width_px == 0
            or abs(_dot(first.direction, other.direction)) < _MIN_LOCAL_ALIGNMENT
        ):
            continue
        for lo, hi in other.intervals:
            if hi - lo < min_support:
                continue
            inside = sum(
                cv2.pointPolygonTest(
                    contour,
                    _advance(other.line.start, other.direction, lo + f * (hi - lo)),
                    True,
                )
                > margin
                for f in (0.25, 0.5, 0.75)
            )
            if inside >= 2:
                return True
    return False


def _boundary_spans(
    start: PixelPoint,
    end: PixelPoint,
    evidence: _MarkingEvidence,
    noise_px: float,
    corner_widths: tuple[float, ...],
) -> tuple[BoundaryEvidence, ...]:
    base = evidence.line.start
    direction = evidence.direction
    positions = tuple(_dot(_subtract(point, base), direction) for point in (start, end))
    (first, first_width), (last, last_width) = sorted(zip(positions, corner_widths))
    cursor = first
    spans = []

    def add(a: float, b: float, observed: bool) -> None:
        spans.append(
            BoundaryEvidence(
                _advance(base, direction, a),
                _advance(base, direction, b),
                "observed" if observed else "estimated",
                (evidence.line.segment_id,),
            )
        )

    for lo, hi in evidence.intervals:
        lo, hi = max(first, lo), min(last, hi)
        if hi <= lo:
            continue
        # Crossing paint explains corner offsets, never holes in the middle.
        tolerance = max(noise_px, first_width) if not spans else noise_px
        if lo > cursor + tolerance:
            add(cursor, lo, False)
            cursor = lo
        if last - hi <= max(noise_px, last_width):
            hi = last
        add(cursor, hi, True)
        cursor = hi
        if cursor >= last:
            break
    if cursor < last - _EPSILON:
        add(cursor, last, False)
    return tuple(spans)


def _unresolved(first: _Anchor, second: _Anchor, reason: str) -> SlotCandidate:
    return SlotCandidate(
        "unresolved", None, (), tuple(sorted({first.node_id, second.node_id})), reason
    )


def _select_nonoverlapping(
    candidates: list[SlotCandidate], width: int, height: int
) -> tuple[SlotCandidate, ...]:
    """Collapse opposite-rail duplicates and withhold conflicting alternatives."""
    selected: list[SlotCandidate] = []
    polygons: list[tuple[NDArray[np.float32], tuple[int, int, int, int], float]] = []
    unresolved: dict[tuple[int, ...], SlotCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.status != "observed",
            item.proposal is None,
            item.junction_ids,
        ),
    ):
        if candidate.proposal is None:
            unresolved.setdefault(candidate.junction_ids, candidate)
            continue
        polygon = candidate.proposal.polygon
        pixels = np.array(
            [
                (point.x * (width - 1), point.y * (height - 1))
                for point in polygon.vertices
            ],
            dtype=np.float32,
        )
        bbox = cv2.boundingRect(pixels)
        area = float(cv2.contourArea(pixels))
        conflict = _overlap_fraction(pixels, bbox, area, polygons)
        if conflict >= 0.80:
            continue
        if conflict > 0.10:
            unresolved.setdefault(
                candidate.junction_ids,
                replace(
                    candidate,
                    status="unresolved",
                    proposal=None,
                    reason="Boundaries conflict with a better-supported proposal",
                ),
            )
            continue
        selected.append(candidate)
        polygons.append((pixels, bbox, area))
    covered_nodes = {
        tuple(pair)
        for candidate in selected
        for pair in _node_pairs(candidate.junction_ids)
    }
    return tuple(selected) + tuple(
        item for ids, item in sorted(unresolved.items()) if ids not in covered_nodes
    )


def _node_pairs(ids: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [ids] + [
        (first, second)
        for index, first in enumerate(ids)
        for second in ids[index + 1 :]
    ]


def _overlap_fraction(
    pixels: NDArray[np.float32],
    bbox: tuple[int, int, int, int],
    area: float,
    others: list[tuple[NDArray[np.float32], tuple[int, int, int, int], float]],
) -> float:
    x, y, width, height = bbox
    largest = 0.0
    for other, (ox, oy, ow, oh), other_area in others:
        if x >= ox + ow or ox >= x + width or y >= oy + oh or oy >= y + height:
            continue
        shared, _ = cv2.intersectConvexConvex(pixels, other)
        largest = max(largest, float(shared) / min(area, other_area))
    return largest


def _interior(point: PixelPoint, width: int, height: int) -> bool:
    x, y = point
    return 2.5 <= x <= width - 3.5 and 2.5 <= y <= height - 3.5


def _convex(points: tuple[PixelPoint, ...]) -> bool:
    edges = [
        _subtract(end, start) for start, end in zip(points, points[1:] + points[:1])
    ]
    crosses = [_cross(a, b) for a, b in zip(edges, edges[1:] + edges[:1])]
    return min(crosses) > _EPSILON or max(crosses) < -_EPSILON


def _subtract(first: PixelPoint, second: PixelPoint) -> PixelPoint:
    return first[0] - second[0], first[1] - second[1]


def _advance(origin: PixelPoint, direction: PixelPoint, distance: float) -> PixelPoint:
    return origin[0] + direction[0] * distance, origin[1] + direction[1] * distance


def _distance(first: PixelPoint, second: PixelPoint) -> float:
    return hypot(*_subtract(first, second))


def _direction(first: PixelPoint, second: PixelPoint) -> PixelPoint:
    x, y = _subtract(second, first)
    length = hypot(x, y)
    return x / length, y / length


def _dot(first: PixelPoint, second: PixelPoint) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _cross(first: PixelPoint, second: PixelPoint) -> float:
    return first[0] * second[1] - first[1] * second[0]

"""Reconstruct local open rows from finite marking ends and junction evidence."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import cos, dist, hypot, radians

from .initialization import BoundaryEvidence, RowEvidence, SlotCandidate
from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .segment_group import SegmentGroup
from .segment_merging import PixelPoint
from .slot_assembly import _MarkingEvidence, _marking_evidence, _supported
from .topology_candidate import TopologyCandidates
from .topology_support import IntersectionSourceSupport

_LOCAL_ALIGNMENT = cos(radians(24))
_TRANSVERSE_ALIGNMENT = cos(radians(30))


@dataclass(frozen=True, slots=True)
class _Endpoint:
    line: LineSegment
    position: PixelPoint
    tip: PixelPoint
    direction: PixelPoint
    width: float
    nodes: tuple[int, ...] = ()
    contexts: tuple[int, ...] = ()
    kind: str = "aligned_endpoint"


def reconstruct_open_rows(
    width: int,
    height: int,
    pairs: tuple[MarkingPair, ...],
    topology: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    source_groups: Mapping[int, SegmentGroup],
    existing_rows: tuple[RowEvidence, ...] = (),
    existing_candidates: tuple[SlotCandidate, ...] = (),
) -> tuple[tuple[RowEvidence, ...], tuple[SlotCandidate, ...]]:
    """Extend RowEvidence without requiring one straight transverse paint axis.

    Finite paired markings must form mutually adjacent, locally coherent paths.
    Independent junction/context evidence corroborates their aligned endpoints.
    These paths establish partial geometry only. Physical extent and the role of
    a curb-like context remain unknown, so this function never creates a polygon.
    """
    scale = max(0.5, hypot(width, height) / 1440)
    endpoints = _endpoints(pairs, topology, support, scale)
    paths = _paths(endpoints, scale)
    known_sets = [
        set(row.separator_ids)
        for row in existing_rows
        if len(row.separator_ids) >= 3 and any(row.compatible_pairs)
    ]
    rows = []
    partials = []
    used: set[int] = set()
    full_pairs = {
        frozenset(candidate.separator_ids)
        for candidate in existing_candidates
        if candidate.proposal is not None
    }
    noise = min(3.0, max(1.25, hypot(width, height) * 0.0018))
    markings = _marking_evidence(topology, pairs, source_groups, noise)
    for path in sorted(paths, key=_path_priority):
        ids = tuple(endpoint.line.segment_id for endpoint in path)
        corroborated = _independent_contexts(path)
        if corroborated < 2:
            continue
        if len(path) < 3 and not (
            len(path) == 2 and all(p.kind == "paint_junction" for p in path)
        ):
            continue
        if any(len(set(ids) & known) >= max(2, 0.6 * len(ids)) for known in known_sets):
            continue
        if len(set(ids) & used) >= 0.5 * len(ids):
            continue
        row_id = len(existing_rows) + len(rows)
        offset = _subtract(path[-1].position, path[0].position)
        length = hypot(*offset)
        direction = offset[0] / length, offset[1] / length
        rows.append(
            RowEvidence(
                None,
                0,
                ids,
                tuple(sorted({node for p in path for node in p.nodes})),
                tuple(p.position for p in path),
                (True,) * (len(path) - 1),
                "Locally ordered marking ends; physical depth and boundary type "
                "require operator review",
                geometry_kind="open_ended",
                anchor_kinds=tuple(p.kind for p in path),
                anchor_line_ids=tuple(p.contexts for p in path),
                separator_tips=tuple(p.tip for p in path),
                direction=direction,
                row_id=row_id,
            )
        )
        used.update(ids)
        for first, second in zip(path, path[1:]):
            if frozenset((first.line.segment_id, second.line.segment_id)) in full_pairs:
                continue
            partials.append(
                SlotCandidate(
                    "unresolved",
                    None,
                    _observations(
                        (first.line.segment_id, second.line.segment_id), markings
                    ),
                    tuple(sorted(set(first.nodes + second.nodes))),
                    "Open row adjacency is supported; finite paint ends do not "
                    "establish the physical depth or an entrance boundary",
                    separator_ids=(first.line.segment_id, second.line.segment_id),
                    anchor_positions=(first.position, second.position),
                    geometry_kind="partial",
                    missing_boundaries=(
                        "physical_depth",
                        "entrance_boundary",
                        "boundary_semantics",
                    ),
                    row_id=row_id,
                )
            )

    # Preserve useful observations for already anchored but incomplete bays.
    for candidate in existing_candidates:
        if candidate.proposal is not None or len(candidate.separator_ids) != 2:
            continue
        if frozenset(candidate.separator_ids) in full_pairs:
            continue
        if not candidate.reason.startswith(
            (
                "Insufficient source fragments",
                "Separator tip is clipped",
                "Rear boundary lacks",
            )
        ):
            continue
        row_id = next(
            (
                i
                for i, row in enumerate(existing_rows)
                if row.rail_id == candidate.rail_id
                and set(candidate.separator_ids) <= set(row.separator_ids)
            ),
            None,
        )
        partials.append(
            replace(
                candidate,
                boundaries=_observations(candidate.separator_ids, markings),
                geometry_kind="partial",
                missing_boundaries=("complete_boundary_extent",),
                row_id=row_id,
            )
        )
    # One pair of source markings is one partial hypothesis. Different local
    # anchor interpretations stay traceable through their nodes and row records.
    unique: dict[frozenset[int], SlotCandidate] = {}
    for candidate in partials:
        key = frozenset(candidate.separator_ids)
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
            continue
        reason = previous.reason
        if candidate.reason not in reason:
            reason += "; " + candidate.reason
        unique[key] = replace(
            previous,
            junction_ids=tuple(
                sorted(set(previous.junction_ids + candidate.junction_ids))
            ),
            missing_boundaries=tuple(
                sorted(set(previous.missing_boundaries + candidate.missing_boundaries))
            ),
            reason=reason,
        )
    # A partial path is not exempt from negative marking evidence merely because
    # its physical depth is unknown. Retain rejected observations for review.
    patterns = [
        _hatched_perimeter(candidate, endpoints, topology, support, markings, scale)
        for candidate in unique.values()
    ]
    for key, candidate in unique.items():
        witness = next(
            (
                pattern
                for pattern in patterns
                if pattern and set(candidate.separator_ids) <= set(pattern[2:])
            ),
            (),
        )
        if witness and len(candidate.anchor_positions) == 2:
            anchors = [
                min(
                    (p for p in endpoints if p.line.segment_id == line_id),
                    key=lambda p: dist(p.position, position),
                )
                for line_id, position in zip(
                    candidate.separator_ids, candidate.anchor_positions
                )
            ]
            if any(set(p.contexts) - set(witness[:2]) for p in anchors):
                witness = ()
        if witness:
            unique[key] = replace(
                candidate,
                rejection_code="non_parking_hatching",
                rejection_line_ids=witness,
                reason="Rejected non-parking hatching: repeated diagonal strokes "
                "terminate on connected perimeter arms; "
                "row anchors are not independent",
            )
    for index, row in enumerate(rows):
        adjacent = [
            unique.get(frozenset(pair))
            for pair in zip(row.separator_ids, row.separator_ids[1:])
        ]
        compatible = tuple(
            row.compatible_pairs[i] if item is None else item.rejection_code is None
            for i, item in enumerate(adjacent)
        )
        if adjacent and all(
            item is not None and item.rejection_code is not None for item in adjacent
        ):
            rows[index] = replace(
                row,
                compatible_pairs=compatible,
                rejection_code="non_parking_hatching",
                reason="Rejected hatched perimeter pattern; see candidate witnesses",
            )
        elif row.compatible_pairs != compatible:
            rows[index] = replace(row, compatible_pairs=compatible)
    return tuple(rows), tuple(unique.values())


def _hatched_perimeter(
    candidate: SlotCandidate,
    endpoints: list[_Endpoint],
    topology: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    markings: Mapping[int, _MarkingEvidence],
    scale: float,
) -> tuple[int, ...]:
    """Recognize a diagonal stroke family anchored around one observed corner.

    Separate T caps or stepped islands are not a connected perimeter corner.
    Parallelism alone is insufficient: both candidate anchors must belong to
    different arms of a source-supported L, with repeated strokes into its wedge.
    This uses the same finite-source support and distinct-stroke semantics as
    slot assembly, without inventing a closing polygon for the partial bay.
    """
    if len(candidate.separator_ids) != 2:
        return ()
    selected = []
    for line_id, position in zip(candidate.separator_ids, candidate.anchor_positions):
        matching = [p for p in endpoints if p.line.segment_id == line_id]
        if not matching:
            return ()
        selected.append(min(matching, key=lambda p: dist(p.position, position)))
    if len(selected) != 2:
        return ()
    first, second = selected
    measured = {(s.candidate_id, s.intersection_index): s for s in support}
    for node in topology.junctions:
        for index, hit in enumerate(node.intersections):
            a, b = hit.line_ids
            if any(set(p.contexts) - {a, b} for p in selected):
                continue
            if hit.topology != "L" or not (
                (a in first.contexts and b in second.contexts)
                or (b in first.contexts and a in second.contexts)
            ):
                continue
            measurement = measured.get((node.candidate_id, index))
            if measurement is None or not all(_supported(s) for s in measurement.lines):
                continue
            if min(dist(p.position, hit.position) for p in selected) < 4 * scale:
                continue
            rays = [_unit(hit.position, p.position) for p in selected]
            turn = _cross(*rays)
            if abs(turn) < 0.4:
                continue
            family = {}
            for endpoint in endpoints:
                if not set(endpoint.contexts) & {a, b}:
                    continue
                if _dot(endpoint.direction, first.direction) < _LOCAL_ALIGNMENT:
                    continue
                # Source-supported strokes must point inside the same wedge.
                if (
                    _cross(rays[0], endpoint.direction) * turn <= 0
                    or _cross(endpoint.direction, rays[1]) * turn <= 0
                ):
                    continue
                evidence = markings[endpoint.line.segment_id]
                if sum(hi - lo for lo, hi in evidence.intervals) < max(
                    24 * scale, 0.25 * min(first.line.length, second.line.length)
                ):
                    continue
                family[endpoint.line.segment_id] = endpoint
            if not set(candidate.separator_ids) <= family.keys():
                continue
            offsets = sorted(
                (_cross(p.position, first.direction), line_id)
                for line_id, p in family.items()
            )
            distinct = []
            margin = max(3 * scale, 2 * max(first.width, second.width))
            for offset, line_id in offsets:
                if not distinct or offset - distinct[-1][0] > margin:
                    distinct.append((offset, line_id))
            gaps = [q[0] - p[0] for p, q in zip(distinct, distinct[1:])]
            if len(distinct) >= 3 and any(
                max(x, y) <= 2.5 * min(x, y) for x, y in zip(gaps, gaps[1:])
            ):
                return (a, b, *(i for _, i in distinct))
    return ()


def _endpoints(
    pairs: tuple[MarkingPair, ...],
    topology: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    scale: float,
) -> list[_Endpoint]:
    paired = {pair.centerline.segment_id: pair for pair in pairs}
    lines = {line.segment_id: line for line in topology.lines}
    measurements = {(s.candidate_id, s.intersection_index): s for s in support}
    hits = defaultdict(list)
    for node in topology.junctions:
        for index, hit in enumerate(node.intersections):
            measured = measurements.get((node.candidate_id, index))
            if measured is None:
                continue
            first, second = (lines[i] for i in hit.line_ids)
            sine = abs(
                _cross(_unit(first.start, first.end), _unit(second.start, second.end))
            )
            if sine < 0.2:
                continue
            widths = [
                min(paired[i].widths_px) if i in paired else 0.0 for i in hit.line_ids
            ]
            if not all(
                _supported(s, crossing / sine)
                for s, crossing in zip(measured.lines, reversed(widths))
            ):
                continue
            for position, line_id in enumerate(hit.line_ids):
                if line_id not in paired:
                    continue
                other_id = hit.line_ids[1 - position]
                kind = "boundary_context"
                if other_id in paired:
                    kind = (
                        "paint_junction"
                        if hit.topology == "T" and hit.roles[position] == "end"
                        else "marking_corner"
                    )
                hits[line_id].append((hit.position, node.candidate_id, other_id, kind))
    endpoints = []
    for pair in pairs:
        line = pair.centerline
        for position, tip in ((line.start, line.end), (line.end, line.start)):
            nearby = [
                hit
                for hit in hits[line.segment_id]
                if dist(position, hit[0]) <= max(14 * scale, 2 * max(pair.widths_px))
            ]
            kinds = [hit[3] for hit in nearby]
            kind = next(
                (
                    k
                    for k in ("paint_junction", "marking_corner", "boundary_context")
                    if k in kinds
                ),
                "aligned_endpoint",
            )
            endpoints.append(
                _Endpoint(
                    line,
                    position,
                    tip,
                    _unit(position, tip),
                    min(pair.widths_px),
                    tuple(sorted({hit[1] for hit in nearby})),
                    tuple(sorted({hit[2] for hit in nearby})),
                    kind,
                )
            )
    return endpoints


def _neighbor_score(first: _Endpoint, second: _Endpoint, scale: float) -> float | None:
    if first.line.segment_id == second.line.segment_id:
        return None
    if _dot(first.direction, second.direction) < _LOCAL_ALIGNMENT:
        return None
    offset = _subtract(second.position, first.position)
    gap = hypot(*offset)
    if (
        not max(8 * scale, 4 * max(first.width, second.width))
        <= gap
        <= (3 * min(first.line.length, second.line.length))
    ):
        return None
    across = offset[0] / gap, offset[1] / gap
    if (
        max(abs(_dot(across, p.direction)) for p in (first, second))
        > _TRANSVERSE_ALIGNMENT
    ):
        return None
    tip_offset = _subtract(second.tip, first.tip)
    tip_gap = hypot(*tip_offset)
    if not 0.45 * gap <= tip_gap <= 2.2 * gap:
        return None
    if _dot(offset, tip_offset) <= 0.4 * gap * tip_gap:
        return None
    return (gap + tip_gap) / 2


def _paths(endpoints: list[_Endpoint], scale: float) -> list[list[_Endpoint]]:
    best = {}
    for i, first in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            second = endpoints[j]
            score = _neighbor_score(first, second, scale)
            if score is None:
                continue
            for a, b in ((i, j), (j, i)):
                offset = _subtract(endpoints[b].position, endpoints[a].position)
                side = 1 if _cross(endpoints[a].direction, offset) > 0 else -1
                key = a, side
                if (score, b) < best.get(key, (float("inf"), -1)):
                    best[key] = score, b
    graph = defaultdict(set)
    for (a, _), (_, b) in best.items():
        if any(best.get((b, side), (None, None))[1] == a for side in (-1, 1)):
            graph[a].add(b)
            graph[b].add(a)
    paths = []
    visited = set()
    for start in sorted(graph, key=lambda i: (len(graph[i]), i)):
        if start in visited or len(graph[start]) != 1:
            continue
        path = []
        current = start
        while current not in visited:
            visited.add(current)
            path.append(endpoints[current])
            following = graph[current] - visited
            if not following:
                break
            current = min(following)
        paths.extend(_split_path(path))
    return paths


def _split_path(path: list[_Endpoint]) -> list[list[_Endpoint]]:
    """Keep smooth local perspective; stop at abrupt turns or missing neighbors."""
    result = []
    run = path[:2]
    for endpoint in path[2:]:
        before = _subtract(run[-1].position, run[-2].position)
        after = _subtract(endpoint.position, run[-1].position)
        first_length, second_length = hypot(*before), hypot(*after)
        if _dot(before, after) < cos(radians(30)) * first_length * second_length or max(
            first_length, second_length
        ) > 2 * min(first_length, second_length):
            result.append(run)
            run = [run[-1]]
        run.append(endpoint)
    result.append(run)
    return result


def _independent_contexts(path: list[_Endpoint]) -> int:
    """Repeated hits on one curb/vehicle edge do not corroborate distinct anchors."""
    groups: list[set[int]] = []
    for endpoint in path:
        if not endpoint.contexts:
            continue
        merged = set(endpoint.contexts)
        remaining = []
        for group in groups:
            if merged & group:
                merged.update(group)
            else:
                remaining.append(group)
        groups = [*remaining, merged]
    return len(groups)


def _path_priority(path: list[_Endpoint]) -> tuple[int, int, tuple[int, ...]]:
    return (
        -sum(bool(p.contexts) for p in path),
        -len(path),
        tuple(p.line.segment_id for p in path),
    )


def _observations(
    ids: tuple[int, ...], markings: Mapping[int, _MarkingEvidence]
) -> tuple[BoundaryEvidence, ...]:
    boundaries = []
    for line_id in ids:
        marking = markings[line_id]
        for lo, hi in marking.intervals:
            start = tuple(
                a + lo * b for a, b in zip(marking.line.start, marking.direction)
            )
            end = tuple(
                a + hi * b for a, b in zip(marking.line.start, marking.direction)
            )
            boundaries.append(BoundaryEvidence(start, end, "observed", (line_id,)))
    return tuple(boundaries)


def _subtract(first: PixelPoint, second: PixelPoint) -> PixelPoint:
    return first[0] - second[0], first[1] - second[1]


def _unit(first: PixelPoint, second: PixelPoint) -> PixelPoint:
    dx, dy = _subtract(second, first)
    length = hypot(dx, dy)
    return dx / length, dy / length


def _dot(first: PixelPoint, second: PixelPoint) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _cross(first: PixelPoint, second: PixelPoint) -> float:
    return first[0] * second[1] - first[1] * second[0]

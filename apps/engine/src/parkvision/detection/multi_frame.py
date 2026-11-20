"""Fuse finite, traceable marking observations, never images or vehicle masks."""

from dataclasses import asdict, dataclass, field, replace
import json
from hashlib import sha256
from math import hypot

import cv2
import numpy as np
from numpy.typing import NDArray

from ..capture.frame import Frame
from .frame_registration import IDENTITY, register_background
from .initialization import GeometryInitialization
from .line_fitting import fit_group_line
from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .opencv_slot_detector import OpenCVSlotDetector
from .segment_group import SegmentGroup
from .segment_merging import (
    angle_difference,
    are_nearly_collinear,
    projected_segment_gap,
    point_line_distance,
)


@dataclass(frozen=True, slots=True)
class GeometryFrame:
    frame: Frame
    source_id: str
    image_id: str
    image_sha256: str | None = None
    registration_points: dict | None = None
    background_mask: NDArray[np.uint8] | None = field(
        default=None, compare=False, repr=False
    )
    validation_mask: NDArray[np.uint8] | None = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True, slots=True)
class MultiFrameInitialization:
    geometry: GeometryInitialization
    frames: tuple[dict, ...]
    observations: tuple[dict, ...]
    markings: tuple[dict, ...]
    conflicts: tuple[dict, ...]
    hypothesis_updates: tuple[dict, ...] = ()


def _transform(line: LineSegment, matrix: np.ndarray, identity: int) -> LineSegment:
    points = np.array([line.start, line.end]) @ matrix[:, :2].T + matrix[:, 2]
    return LineSegment(identity, tuple(points[0]), tuple(points[1]))


def _same(first: LineSegment, second: LineSegment, distance: float, gap: float) -> bool:
    return are_nearly_collinear(first, second, 3.0, distance) and (
        projected_segment_gap(first, second) <= gap
    )


def _overlap(first, second) -> float:
    a = np.float32([(p.x, p.y) for p in first.proposal.polygon.vertices])
    b = np.float32([(p.x, p.y) for p in second.proposal.polygon.vertices])
    # Near-coincident float32 edges can make intersectConvexConvex report zero.
    # Bidirectional vertex proximity catches only subpixel duplicate geometry.
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    if max(distances.min(axis=0).max(), distances.min(axis=1).max()) < 1e-4:
        return 1.0
    area, _ = cv2.intersectConvexConvex(a, b)
    return area / max(1e-12, min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b))))


def _internal_separator(candidate, pair, reference_lines, width, height) -> bool:
    separators = [
        line for line in reference_lines if line.segment_id in candidate.separator_ids
    ]
    axis = pair.centerline
    if not any(
        angle_difference(axis, line) <= 12 and axis.length >= 0.7 * line.length
        for line in separators
    ):
        return False
    polygon = np.float32(
        [
            (p.x * (width - 1), p.y * (height - 1))
            for p in candidate.proposal.polygon.vertices
        ]
    )
    midpoint = tuple((np.array(axis.start) + axis.end) / 2)
    return cv2.pointPolygonTest(polygon, midpoint, True) > max(
        3, 2 * max(pair.widths_px)
    )


def _observed_bridge(
    pair: MarkingPair, clusters: list[dict], sources: dict, scale: float
) -> bool:
    """Only an actually covered new stripe may connect separate old fragments."""
    axis = pair.centerline
    direction = (np.array(axis.end) - axis.start) / axis.length

    def interval(line):
        return sorted(
            float(np.dot(np.array(p) - axis.start, direction))
            for p in (line.start, line.end)
        )

    intervals = sorted(interval(c["pair"].centerline) for c in clusters)
    # LSD terminates at transverse painted intersections. Permit only a local
    # stripe-width interruption, not the long formerly occluded interval.
    allowed_gap = max(3 * scale, max(pair.widths_px) + 4 * scale)
    for (_, left), (right, _) in zip(intervals, intervals[1:]):
        # Competing overlapping axes stay ambiguous. Both original LSD edge
        # groups must cover the gap between separate collinear fragments.
        if right < left - 2 * scale:
            return False
        for edge in pair.edges:
            cursor = left
            for start, end in sorted(
                interval(s) for s in sources[edge.segment_id].segments
            ):
                if end < cursor:
                    continue
                if start > cursor + allowed_gap:
                    break
                cursor = max(cursor, end)
            if cursor < right - 3 * scale:
                return False
    return True


def analyze_frames(
    frames: tuple[GeometryFrame, ...],
    *,
    background_mask: NDArray[np.uint8] | None = None,
    validation_mask: NDArray[np.uint8] | None = None,
    detector: OpenCVSlotDetector | None = None,
) -> MultiFrameInitialization:
    """Reference is first. Extra views fail closed and cannot erase its full ROI.

    Distinct source/camera IDs are rejected before image analysis. A declared
    stationary background mask is mandatory for nonidentical source images.
    Matching uses complete-link axis agreement, finite overlap/gap and stripe
    width; different parallel separators are never grouped by direction alone.
    Reassembly retains the original source coverage tests and open-row logic.
    """
    if not frames:
        raise ValueError("At least one geometry frame is required")
    if any(
        not item.source_id or not item.image_id or not item.frame.camera_id
        for item in frames
    ):
        raise ValueError("Source, camera and image identifiers must be nonempty")
    if len({item.image_id for item in frames}) != len(frames):
        raise ValueError("Image identifiers must be unique")
    detector = detector or OpenCVSlotDetector()
    reference = frames[0]
    image = reference.frame.image
    base = detector.analyze(reference.frame)
    height, width = image.shape[:2]
    scale = max(0.5, hypot(width, height) / 1440)
    records, observations, conflicts = [], [], []
    # Each cluster retains individual axes to prevent transitive drift.
    clusters: list[dict] = []
    sources: dict[int, SegmentGroup] = {}
    segments, context = [], []
    context_axes: dict[int, list[LineSegment]] = {}
    next_id = 0
    accepted_images = []
    accepted_hashes = set()
    aliases: dict[int, int] = {}
    for index, item in enumerate(frames):
        pixels = item.frame.image
        pixel_hash = sha256(pixels.tobytes()).hexdigest()
        record = {
            "frame_index": index,
            "image_id": item.image_id,
            "camera_id": item.frame.camera_id,
            "source_id": item.source_id,
            "image_sha256": item.image_sha256,
            "pixel_sha256": pixel_hash,
            "accepted": False,
            "reason": "reference",
            "transform": None,
        }
        records.append(record)
        if (
            item.source_id != reference.source_id
            or item.frame.camera_id != reference.frame.camera_id
        ):
            record["reason"] = "source_or_camera_mismatch"
            continue
        if pixels.shape != image.shape:
            record["reason"] = "resolution_mismatch"
            continue
        if pixel_hash in accepted_hashes:
            record.update(reason="duplicate_pixels", transform=IDENTITY)
            continue
        if index:
            options = {}
            fit_mask = (
                item.background_mask
                if item.background_mask is not None
                else background_mask
            )
            check_mask = (
                item.validation_mask
                if item.validation_mask is not None
                else validation_mask
            )
            if check_mask is not None:
                options["validation_mask"] = check_mask
            if item.registration_points is not None:
                options["control_points"] = item.registration_points
                record["registration_points_input_sha256"] = sha256(
                    json.dumps(item.registration_points, sort_keys=True).encode()
                ).hexdigest()
            record["registration_mask_sha256"] = (
                sha256(fit_mask.tobytes()).hexdigest() if fit_mask is not None else None
            )
            record["validation_mask_sha256"] = (
                sha256(check_mask.tobytes()).hexdigest()
                if check_mask is not None
                else None
            )
            registration = register_background(image, pixels, fit_mask, **options)
            record["registration"] = asdict(registration)
            record["reason"] = registration.reason
            record["transform"] = registration.matrix
            if not registration.accepted:
                continue
            matrix = np.array(registration.matrix)
            aligned = cv2.warpAffine(pixels, matrix, (width, height))
            # Compare full views after alignment; a duplicate has no new evidence.
            small = cv2.resize(aligned, (160, 120)).astype(float)
            if np.max(cv2.absdiff(aligned, image)) <= 3 and any(
                np.mean(np.abs(small - prior)) < 1.5 for prior in accepted_images
            ):
                record["reason"] = "near_duplicate_no_new_evidence"
                continue
        else:
            matrix = np.array(IDENTITY)
            aligned = pixels
            record["transform"] = IDENTITY
        record["accepted"] = True
        accepted_images.append(cv2.resize(aligned, (160, 120)).astype(float))
        accepted_hashes.add(pixel_hash)
        result = base if index == 0 else detector.analyze(item.frame)
        record["single_frame"] = {
            "separators": len(result.pairs),
            "corners": len(result.topology.junctions),
            "full_roi": sum(c.proposal is not None for c in result.candidates),
            "partial": sum(
                c.geometry_kind == "partial" and c.rejection_code is None
                for c in result.candidates
            ),
        }
        record["source_topology"] = asdict(result.topology)
        # Preserve reference IDs so baseline candidates remain valid after fusion.
        all_lines = {
            line.segment_id: line for line in result.segments + result.fitted_lines
        }
        for pair in result.pairs:
            for line in (*pair.edges, pair.centerline):
                all_lines[line.segment_id] = line
        ids = {
            old: old if index == 0 else next_id + n
            for n, old in enumerate(sorted(all_lines))
        }
        next_id = max(next_id, max(ids.values(), default=-1) + 1)
        transformed = {
            old: _transform(line, matrix, ids[old]) for old, line in all_lines.items()
        }
        local_sources = {
            ids[line_id]: SegmentGroup(
                tuple(transformed[s.segment_id] for s in group.segments)
            )
            for line_id, group in result.source_groups
        }
        sources.update(local_sources)
        segments.extend(transformed[s.segment_id] for s in result.segments)
        for pair in result.pairs:
            transformed_pair = MarkingPair(
                tuple(transformed[e.segment_id] for e in pair.edges),
                transformed[pair.centerline.segment_id],
                pair.widths_px,
                pair.overlap_ratio,
            )
            axis = transformed_pair.centerline
            paint_width = sum(pair.widths_px) / 2
            observation_id = len(observations)
            observations.append(
                {
                    "observation_id": observation_id,
                    "frame_index": index,
                    "kind": "paired_marking",
                    "source_axis": asdict(pair.centerline),
                    "reference_axis": asdict(axis),
                    "source_edges": [asdict(e) for e in pair.edges],
                    "source_lsd": [
                        asdict(s)
                        for e in pair.edges
                        for s in dict(result.source_groups)[e.segment_id].segments
                    ],
                    "reference_lsd_ids": [
                        s.segment_id
                        for e in transformed_pair.edges
                        for s in sources[e.segment_id].segments
                    ],
                    "quality": "geometric_stripe_not_semantic_confirmation",
                }
            )
            matches = [
                cluster
                for cluster in clusters
                if all(
                    _same(
                        axis,
                        prior,
                        max(1.5 * scale, 0.35 * min(paint_width, cluster["width"])),
                        min(28 * scale, 0.2 * min(axis.length, prior.length)),
                    )
                    for prior in cluster["axes"]
                )
                and 0.65 < paint_width / cluster["width"] < 1.55
            ]
            if index == 0:
                matches = []
            same_frame = any(index in cluster["frames"] for cluster in matches)
            if (
                len(matches) > 1
                and not same_frame
                and _observed_bridge(transformed_pair, matches, local_sources, scale)
            ):
                canonical = matches[0]
                for redundant in matches[1:]:
                    old_pair = redundant["pair"]
                    aliases[old_pair.centerline.segment_id] = canonical[
                        "pair"
                    ].centerline.segment_id
                    old_groups = redundant["edge_groups"]
                    old_edges = old_pair.edges
                    if point_line_distance(
                        old_edges[1].start, canonical["pair"].edges[0]
                    ) < point_line_distance(
                        old_edges[0].start, canonical["pair"].edges[0]
                    ):
                        old_groups = old_groups[::-1]
                        old_edges = old_edges[::-1]
                    for n in range(2):
                        aliases[old_edges[n].segment_id] = (
                            canonical["pair"].edges[n].segment_id
                        )
                        canonical["edge_groups"][n].extend(old_groups[n])
                    canonical["axes"].extend(redundant["axes"])
                    canonical["frames"] = sorted(
                        set(canonical["frames"] + redundant["frames"])
                    )
                    canonical["observations"].extend(redundant["observations"])
                    clusters.remove(redundant)
                matches = [canonical]
            if len(matches) > 1 or same_frame:
                conflicts.append(
                    {
                        "observation_id": observation_id,
                        "reason": "ambiguous_or_same_frame_match",
                    }
                )
                continue
            if matches:
                cluster = matches[0]
                # Opposite LSD endpoint order must not exchange stripe edges.
                current = cluster["pair"]
                edges = transformed_pair.edges
                if point_line_distance(
                    edges[1].start, current.edges[0]
                ) < point_line_distance(edges[0].start, current.edges[0]):
                    edges = edges[::-1]
                cluster["edge_groups"][0].extend(
                    local_sources[edges[0].segment_id].segments
                )
                cluster["edge_groups"][1].extend(
                    local_sources[edges[1].segment_id].segments
                )
                cluster["axes"].append(axis)
                cluster["frames"] = sorted(set(cluster["frames"] + [index]))
                cluster["observations"].append(observation_id)
            else:
                # A shifted near-overlapping stripe is a conflict, not extra support.
                nearby = [
                    c
                    for c in clusters
                    if index not in c["frames"]
                    and angle_difference(axis, c["axes"][0]) < 8
                    and are_nearly_collinear(
                        axis, c["axes"][0], 8, max(4 * scale, paint_width + c["width"])
                    )
                    and projected_segment_gap(axis, c["axes"][0]) == 0
                ]
                if nearby:
                    conflicts.append(
                        {
                            "observation_id": observation_id,
                            "reason": "inconsistent_stripe_position_or_width",
                        }
                    )
                    continue
                clusters.append(
                    {
                        "pair": transformed_pair,
                        "axes": [axis],
                        "frames": [index],
                        "observations": [observation_id],
                        "width": paint_width,
                        "edge_groups": [
                            list(local_sources[e.segment_id].segments)
                            for e in transformed_pair.edges
                        ],
                    }
                )
        paired_ids = {e.segment_id for pair in result.pairs for e in pair.edges}
        for line in result.fitted_lines:
            if line.segment_id in paired_ids:
                continue
            measured = transformed[line.segment_id]
            match = next(
                (
                    prior
                    for prior in context
                    if all(
                        _same(measured, axis, 1.5 * scale, 12 * scale)
                        for axis in context_axes[prior.segment_id]
                    )
                ),
                None,
            )
            target_id = measured.segment_id if match is None else match.segment_id
            observations.append(
                {
                    "observation_id": len(observations),
                    "frame_index": index,
                    "kind": "unpaired_context",
                    "source_axis": asdict(line),
                    "reference_axis": asdict(measured),
                    "fusion_line_id": target_id,
                    "source_lsd": [
                        asdict(s)
                        for s in dict(result.source_groups)[line.segment_id].segments
                    ],
                    "reference_lsd_ids": [
                        s.segment_id
                        for s in local_sources[measured.segment_id].segments
                    ],
                    "quality": "context_only_not_confirmed_paint_or_curb",
                }
            )
            if match is None:
                context.append(measured)
                context_axes[measured.segment_id] = [measured]
            else:
                context_axes[match.segment_id].append(measured)
                # Preserve finite fragments for support without inventing a bridge.
                sources[match.segment_id] = SegmentGroup(
                    sources[match.segment_id].segments
                    + local_sources[measured.segment_id].segments
                )
                merged = fit_group_line(sources[match.segment_id], segment_id=next_id)
                next_id += 1
                context[context.index(match)] = replace(
                    merged, segment_id=match.segment_id
                )
    if sum(record["accepted"] for record in records) == 1:
        geometry = base
    else:
        for old in aliases:
            while aliases[old] in aliases:
                aliases[old] = aliases[aliases[old]]
        pairs = []
        for cluster in clusters:
            pair = cluster["pair"]
            if len(cluster["axes"]) > 1:
                # A fitted envelope may span gaps; assembly still measures raw coverage.
                edges = []
                for original, group in zip(
                    pair.edges, cluster["edge_groups"], strict=True
                ):
                    edge = fit_group_line(
                        SegmentGroup(tuple(group)), segment_id=next_id
                    )
                    next_id += 1
                    edge = replace(edge, segment_id=original.segment_id)
                    sources[edge.segment_id] = SegmentGroup(tuple(group))
                    edges.append(edge)
                axis = fit_group_line(
                    SegmentGroup(tuple(cluster["axes"])), segment_id=next_id
                )
                next_id += 1
                axis = replace(axis, segment_id=pair.centerline.segment_id)
                pair = MarkingPair(
                    tuple(edges), axis, pair.widths_px, pair.overlap_ratio
                )
            cluster["output_id"] = pair.centerline.segment_id
            pairs.append(pair)
        fitted = tuple(e for pair in pairs for e in pair.edges) + tuple(context)
        active_sources = {line.segment_id: sources[line.segment_id] for line in fitted}
        geometry = detector.assemble_evidence(
            reference.frame, tuple(segments), fitted, active_sources, tuple(pairs)
        )
        baseline_full = tuple(
            replace(
                c,
                separator_ids=tuple(aliases.get(i, i) for i in c.separator_ids),
                rail_id=aliases.get(c.rail_id, c.rail_id),
                boundaries=tuple(
                    replace(b, line_ids=tuple(aliases.get(i, i) for i in b.line_ids))
                    for b in c.boundaries
                ),
                junction_ids=tuple(
                    node.candidate_id
                    for node in geometry.topology.junctions
                    if any(
                        hypot(
                            node.position[0] - old.position[0],
                            node.position[1] - old.position[1],
                        )
                        < 5 * scale
                        for old in base.topology.junctions
                        if old.candidate_id in c.junction_ids
                    )
                ),
            )
            for c in base.candidates
            if c.proposal is not None
        )
        protected, disputed = [], []
        new_pairs = [
            pair
            for pair, cluster in zip(pairs, clusters, strict=True)
            if 0 not in cluster["frames"]
        ]
        for candidate in baseline_full:
            internal = [
                pair
                for pair in new_pairs
                if _internal_separator(
                    candidate, pair, base.topology.lines, width, height
                )
            ]
            if internal:
                conflicts.append(
                    {
                        "candidate_separator_ids": candidate.separator_ids,
                        "line_ids": [pair.centerline.segment_id for pair in internal],
                        "reason": "reference_draft_has_new_internal_separator",
                    }
                )
                disputed.append(
                    replace(
                        candidate,
                        status="unresolved",
                        proposal=None,
                        reason=(
                            "New source-supported internal separator conflicts "
                            "with reference draft"
                        ),
                    )
                )
            else:
                protected.append(candidate)
        baseline_full = tuple(protected)
        full_keys = {frozenset(c.separator_ids) for c in baseline_full}
        known_rejections = {
            frozenset(aliases.get(i, i) for i in c.separator_ids): c
            for c in base.candidates
            if c.rejection_code
        }
        additions = list(disputed)
        for candidate in geometry.candidates:
            prior = known_rejections.get(frozenset(candidate.separator_ids))
            if prior is not None:
                candidate = replace(
                    candidate,
                    status="unresolved",
                    proposal=None,
                    geometry_kind="partial",
                    rejection_code=prior.rejection_code,
                    rejection_line_ids=tuple(
                        aliases.get(i, i) for i in prior.rejection_line_ids
                    ),
                    reason=prior.reason,
                )
            if (
                candidate.proposal is None
                and frozenset(candidate.separator_ids) in full_keys
            ):
                continue
            if candidate.proposal is not None and any(
                _overlap(candidate, prior) > 0.1 for prior in baseline_full
            ):
                continue
            additions.append(candidate)
        represented = {
            frozenset(c.separator_ids) for c in baseline_full + tuple(additions)
        }
        for candidate in base.candidates:
            key = frozenset(aliases.get(i, i) for i in candidate.separator_ids)
            if candidate.geometry_kind == "partial" and key not in represented:
                additions.append(
                    replace(
                        candidate,
                        separator_ids=tuple(
                            aliases.get(i, i) for i in candidate.separator_ids
                        ),
                        rail_id=aliases.get(candidate.rail_id, candidate.rail_id),
                        boundaries=tuple(
                            replace(
                                b, line_ids=tuple(aliases.get(i, i) for i in b.line_ids)
                            )
                            for b in candidate.boundaries
                        ),
                        row_id=None,
                        junction_ids=(),
                        reason="Retained unresolved reference evidence. "
                        + candidate.reason,
                    )
                )
        # Keep the reference coordinate system and its full, source-supported drafts.
        geometry = replace(geometry, candidates=baseline_full + tuple(additions))
    markings = tuple(
        {
            "line_id": c.get("output_id", c["pair"].centerline.segment_id),
            "frame_indices": c["frames"],
            "observation_ids": c["observations"],
            "support_kind": (
                "multiple_frames" if len(c["frames"]) > 1 else "single_frame"
            ),
            "visibility_kind": (
                "complementary_finite_spans_occlusion_not_inferred"
                if len(c["axes"]) > 1
                and any(
                    abs(axis.length - c["axes"][0].length) > 5 * scale
                    or hypot(
                        axis.start[0] - c["axes"][0].start[0],
                        axis.start[1] - c["axes"][0].start[1],
                    )
                    > 5 * scale
                    for axis in c["axes"][1:]
                )
                else "observed_finite_spans"
            ),
            "not_detected_in_frames": [
                r["frame_index"]
                for r in records
                if r["accepted"] and r["frame_index"] not in c["frames"]
            ],
            "visibility": "finite_observed_spans_only_missing_detection_is_unknown",
        }
        for c in clusters
    )
    for marking in markings:
        for observation_id in marking["observation_ids"]:
            observations[observation_id]["fusion_line_id"] = marking["line_id"]
    updates = []
    for candidate_id, candidate in enumerate(base.candidates, 1):
        if candidate.geometry_kind != "partial":
            continue
        key = frozenset(aliases.get(i, i) for i in candidate.separator_ids)
        matching = [c for c in geometry.candidates if frozenset(c.separator_ids) == key]
        closed = any(c.proposal is not None for c in matching)
        updates.append(
            {
                "reference_candidate_id": candidate_id,
                "separator_ids": sorted(key),
                "state": (
                    "rejected_non_parking"
                    if any(c.rejection_code for c in matching)
                    else "closed_with_source_support" if closed else "still_unresolved"
                ),
                "rejection_codes": [
                    c.rejection_code for c in matching if c.rejection_code
                ],
                "missing_before": candidate.missing_boundaries,
                "missing_after": [c.missing_boundaries for c in matching],
                "reasons": [c.reason for c in matching],
            }
        )
    return MultiFrameInitialization(
        geometry,
        tuple(records),
        tuple(observations),
        markings,
        tuple(conflicts),
        tuple(updates),
    )

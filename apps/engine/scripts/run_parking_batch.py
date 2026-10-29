import argparse
import datetime
import json
import re
from colorsys import hsv_to_rgb
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from parkvision.capture.frame import Frame
from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_pairing import pair_marking_edges
from parkvision.detection.marking_topology import find_topology_candidates
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import group_segments
from parkvision.detection.topology_candidate import TopologyCandidates
from parkvision.detection.topology_support import IntersectionSourceSupport
from parkvision.detection.topology_support_measurement import measure_topology_support


def image_sort_key(path: Path) -> list[str | int]:
    """Sort image names naturally, so image-2 precedes image-10."""
    return [
        int(part) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def render_groups(
    image: NDArray[np.uint8], groups: tuple[SegmentGroup, ...]
) -> NDArray[np.uint8]:
    """Draw the unchanged source fragments without connecting gaps within groups."""
    preview = image.copy()
    multi_count = sum(len(group.segments) > 1 for group in groups)
    colors = []
    for index in range(multi_count):
        red, green, blue = hsv_to_rgb(index / multi_count, 1, 1)
        colors.append((round(blue * 255), round(green * 255), round(red * 255)))
    Random(0).shuffle(colors)
    color_index = 0
    # Draw singletons first so gray does not cover colored groups at crossings.
    for group in sorted(groups, key=lambda group: len(group.segments) > 1):
        color = (128, 128, 128)
        if len(group.segments) > 1:
            color = colors[color_index]
            color_index += 1
        for segment in group.segments:
            # Fixed-point coordinates retain subpixel endpoints without joining gaps.
            start = tuple(round(value * 256) for value in segment.start)
            end = tuple(round(value * 256) for value in segment.end)
            cv2.line(preview, start, end, color, 1, cv2.LINE_AA, shift=8)
    return preview


def render_fitted_lines(
    image: NDArray[np.uint8], fitted_lines: tuple[LineSegment, ...]
) -> NDArray[np.uint8]:
    """Show fitted extents separately from the gap-preserving source preview."""
    preview = image.copy()
    for line in fitted_lines:
        start = tuple(round(value * 256) for value in line.start)
        end = tuple(round(value * 256) for value in line.end)
        cv2.line(preview, start, end, (0, 255, 255), 1, cv2.LINE_AA, shift=8)
    cv2.putText(
        preview,
        "Fitted extents (may span gaps)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def render_marking_axes(
    image: NDArray[np.uint8],
    fitted_lines: tuple[LineSegment, ...],
    pairs: tuple[MarkingPair, ...],
) -> NDArray[np.uint8]:
    """Show geometric hypotheses, retaining unmatched extents in gray."""
    preview = image.copy()
    paired_ids = {edge.segment_id for pair in pairs for edge in pair.edges}
    for line in fitted_lines:
        color = (255, 255, 0) if line.segment_id in paired_ids else (128, 128, 128)
        start = tuple(round(value * 256) for value in line.start)
        end = tuple(round(value * 256) for value in line.end)
        cv2.line(preview, start, end, color, 1, cv2.LINE_AA, shift=8)
    for pair in pairs:
        start = tuple(round(value * 256) for value in pair.centerline.start)
        end = tuple(round(value * 256) for value in pair.centerline.end)
        cv2.line(preview, start, end, (0, 255, 0), 2, cv2.LINE_AA, shift=8)
    cv2.putText(
        preview,
        "Axes: green; paired: cyan; unpaired: gray (extents may span gaps)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return preview


def line_data(line: LineSegment) -> dict[str, object]:
    """Serialize a line's ID, unrounded endpoints and derived pixel length."""
    return {
        "segment_id": line.segment_id,
        "start": line.start,
        "end": line.end,
        "length": line.length,
    }


def render_topology_candidates(
    image: NDArray[np.uint8], result: TopologyCandidates
) -> NDArray[np.uint8]:
    """Show candidates only; orange = extension, violet = unpaired context."""
    preview = image.copy()
    context_ids = set(result.unpaired_line_ids)
    for line in result.lines:
        color = (128, 128, 128) if line.segment_id in context_ids else (0, 255, 0)
        start = tuple(round(value * 256) for value in line.start)
        end = tuple(round(value * 256) for value in line.end)
        cv2.line(preview, start, end, color, 1, cv2.LINE_AA, shift=8)
    for endpoint in result.endpoints:
        # All context ends are in JSON; marking every tiny edge obscures the image.
        if endpoint.line_id not in context_ids:
            point = tuple(round(value * 256) for value in endpoint.position)
            cv2.circle(preview, point, 2 * 256, (255, 255, 0), 1, cv2.LINE_AA, shift=8)
    height, width = image.shape[:2]
    for node in result.junctions:
        x, y = (round(value) for value in node.position)
        if not (0 <= x < width and 0 <= y < height):
            continue  # Off-frame hypotheses are still serialized without clipping.
        labels = "/".join(sorted({hit.topology for hit in node.intersections}))
        if all(hit.hypothetical for hit in node.intersections):
            color = (0, 165, 255)
        elif all(hit.uses_unpaired_context for hit in node.intersections):
            color = (255, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.circle(preview, (x, y), 5, color, 1, cv2.LINE_AA)
        cv2.putText(
            preview,
            f"{node.candidate_id}:{labels}?",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    for row, (label, color) in enumerate(
        (
            ("L/T/X candidates only; source paint coverage NOT checked", (0, 255, 255)),
            ("Nodes: yellow=axes; violet=context; orange=extension", (0, 165, 255)),
            (
                "Cyan dots: axis endpoints (all endpoints retained in JSON)",
                (255, 255, 0),
            ),
        )
    ):
        cv2.putText(
            preview,
            label,
            (10, 22 + row * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return preview


def save_topology_diagnostics(
    image: NDArray[np.uint8],
    image_path: Path,
    output_dir: Path,
    fitted_lines: tuple[LineSegment, ...],
    pairs: tuple[MarkingPair, ...],
    parameters: dict[str, float],
) -> TopologyCandidates:
    """Use the public API and link to existing, unchanged provenance files."""
    # Paired edges are represented by their axis; only unmatched edges add context.
    unpaired = _unpaired_lines(fitted_lines, pairs)
    started = perf_counter()
    result = find_topology_candidates(pairs, unpaired, **parameters)
    elapsed = perf_counter() - started
    # Save all evidence in JSON; the preview intentionally shows fewer endpoint markers.
    data = topology_data(image_path, fitted_lines, pairs, result, parameters)
    _write_json(output_dir / "topology.json", data, allow_nan=False)
    path = output_dir / "preview_topology_candidates.png"
    _write_preview(path, render_topology_candidates(image, result))
    print(
        f"  Topology candidates: {len(result.junctions)} nodes, "
        f"{len(result.endpoints)} endpoints; geometry: {elapsed:.3f} s; {path.name}"
    )
    return result


def save_topology_support_diagnostics(
    image: NDArray[np.uint8],
    output_dir: Path,
    result: TopologyCandidates,
    pairs: tuple[MarkingPair, ...],
    source_groups: dict[int, SegmentGroup],
    parameters: dict[str, float],
) -> None:
    """Add two opt-in files; preserve the nine existing diagnostics byte for byte."""
    # 1. Measure support using the existing topology and unchanged source groups.
    measurements = measure_topology_support(result, pairs, source_groups, **parameters)

    # 2. Save each exact intersection, including its source IDs and window coverage.
    intersection_records = [
        _intersection_support_data(measurement) for measurement in measurements
    ]
    _write_json(
        output_dir / "topology_support.json",
        {
            "schema_version": 1,
            "coordinate_system": "pixels",
            "interpretation": (
                "longitudinal raw LSD interval support only; NOT paint confirmation; "
                "both signed arms retained, windows not clipped to fitted extents; "
                "no candidate or line is filtered"
            ),
            "topology_file": "topology.json",
            "provenance_file": "markings.json",
            "source_segments_file": "segments.json",
            "parameters": parameters,
            "intersections": intersection_records,
        },
        allow_nan=False,
    )

    # 3. Render a separate preview after saving the machine-readable evidence.
    preview = render_topology_support(image, result, measurements, source_groups)
    _write_preview(
        output_dir / "preview_topology_support.png",
        preview,
    )
    print(f"  Source support: {len(measurements)} pairwise hits; topology_support.json")


def _intersection_support_data(
    measurement: IntersectionSourceSupport,
) -> dict[str, object]:
    """Serialize one measurement and include the computed coverage properties."""
    intersection_record = asdict(measurement)
    for line_support, line_record in zip(
        measurement.lines, intersection_record["lines"]
    ):
        for edge_support, edge_record in zip(line_support.edges, line_record["edges"]):
            # asdict() handles stored fields; properties need to be added explicitly.
            windows = (
                ("junction", edge_support.junction),
                ("before", edge_support.before),
                ("after", edge_support.after),
            )
            for window_name, window_support in windows:
                window_record = edge_record[window_name]
                window_record["coverage_ratio"] = window_support.coverage_ratio
    return intersection_record


def render_topology_support(
    image: NDArray[np.uint8],
    result: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
    source_groups: dict[int, SegmentGroup],
) -> NDArray[np.uint8]:
    """Show unchanged raw fragments participating in local windows, never axes."""
    # 1. Select the raw fragments that contribute to any of the measured windows.
    preview = image.copy()
    supporting_source_ids = _supporting_source_ids(support)

    # 2. Draw their original endpoints, preserving gaps between separate fragments.
    for group in source_groups.values():
        for segment in group.segments:
            if segment.segment_id not in supporting_source_ids:
                continue
            start_point = tuple(round(value * 256) for value in segment.start)
            end_point = tuple(round(value * 256) for value in segment.end)
            cv2.line(
                preview,
                start_point,
                end_point,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
                shift=8,
            )

    # 3. Label each exact intersection, even when several share one junction node.
    _draw_support_intersections(preview, result, support)

    # 4. Explain how raw fragments and per-edge coverage are represented.
    for row, label in enumerate(
        (
            "Yellow: raw LSD fragments overlapping local windows; no paint test",
            "J%: longitudinal coverage per edge, grouped by line; arms in JSON",
        )
    ):
        cv2.putText(
            preview,
            label,
            (10, 22 + 20 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def _draw_support_intersections(
    preview: NDArray[np.uint8],
    result: TopologyCandidates,
    support: tuple[IntersectionSourceSupport, ...],
) -> None:
    """Draw exact hit positions and their per-edge junction coverage in place."""
    junctions_by_id = {junction.candidate_id: junction for junction in result.junctions}
    for measurement in support:
        junction = junctions_by_id[measurement.candidate_id]
        intersection = junction.intersections[measurement.intersection_index]
        x, y = (round(value) for value in intersection.position)
        if not (0 <= x < preview.shape[1] and 0 <= y < preview.shape[0]):
            continue

        # '/' separates source edges of one line; '|' separates the two lines.
        line_coverages = []
        for line_support in measurement.lines:
            edge_percentages = [
                f"{edge.junction.coverage_ratio * 100:.0f}"
                for edge in line_support.edges
            ]
            line_coverages.append("/".join(edge_percentages))
        coverage_label = "|".join(line_coverages)
        label = (
            f"{measurement.candidate_id}:{measurement.intersection_index} "
            f"J[{coverage_label}]%"
        )
        label_position = (x + 6, y - 6 - 14 * measurement.intersection_index)

        cv2.circle(preview, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            preview,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _supporting_source_ids(
    measurements: tuple[IntersectionSourceSupport, ...],
) -> set[int]:
    """Collect contributing raw LSD IDs from the junction and both signed arms."""
    source_ids: set[int] = set()
    for measurement in measurements:
        for line_support in measurement.lines:
            for edge_support in line_support.edges:
                source_ids.update(edge_support.junction.source_segment_ids)
                source_ids.update(edge_support.before.source_segment_ids)
                source_ids.update(edge_support.after.source_segment_ids)
    return source_ids


def topology_data(
    image_path: Path,
    fitted_lines: tuple[LineSegment, ...],
    pairs: tuple[MarkingPair, ...],
    result: TopologyCandidates,
    parameters: dict[str, float],
) -> dict[str, object]:
    """Serialize candidate geometry separately from detection, rendering and I/O."""
    group_indices = {line.segment_id: i for i, line in enumerate(fitted_lines)}
    pair_by_id = {pair.centerline.segment_id: pair for pair in pairs}
    return {
        "schema_version": 1,
        "source": str(image_path),
        "coordinate_system": "pixels",
        "interpretation": (
            "pairwise geometric candidates; fitted extents may span gaps; "
            "source paint coverage NOT checked; unpaired edges are context only"
        ),
        "provenance_file": "markings.json",
        "source_segments_file": "segments.json",
        "parameters": parameters,
        "lines": [
            _topology_line_data(line, pair_by_id, group_indices)
            for line in result.lines
        ],
        "junctions": [
            {
                "candidate_id": node.candidate_id,
                "position": node.position,
                "intersections": [
                    {**asdict(hit), "hypothetical": hit.hypothetical}
                    for hit in node.intersections
                ],
            }
            for node in result.junctions
        ],
        "endpoints": [asdict(endpoint) for endpoint in result.endpoints],
    }


def _topology_line_data(
    line: LineSegment,
    pair_by_id: dict[int, MarkingPair],
    group_indices: dict[int, int],
) -> dict[str, object]:
    """Link one axis or unmatched edge to the existing source-group records."""
    pair = pair_by_id.get(line.segment_id)
    if pair is None:
        kind = "unpaired_context"
        pair_id = None
        source_groups = [group_indices[line.segment_id]]
    else:
        kind = "axis"
        pair_id = line.segment_id
        source_groups = [group_indices[edge.segment_id] for edge in pair.edges]
    return {
        **line_data(line),
        "kind": kind,
        "pair_centerline_id": pair_id,
        "group_indices": source_groups,
    }


def parse_arguments() -> (
    tuple[
        argparse.Namespace, dict[str, float], dict[str, float], dict[str, float] | None
    ]
):
    """Read and validate all options before accessing images or writing outputs."""
    parser = argparse.ArgumentParser(
        description="Save LSD, groups, fitted extents and marking-axis hypotheses."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples",
        help="Input image or folder containing images (default: engine/samples).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder; use a location outside the repository.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=30.0,
        help="Minimum length for the LSD preview only (default: 30 px).",
    )
    parser.add_argument(
        "--marking-min-width",
        type=float,
        default=2.0,
        help="Minimum paint-edge separation at both shared ends (default: 2 px).",
    )
    parser.add_argument(
        "--marking-max-width",
        type=float,
        default=12.0,
        help="Maximum paint-edge separation at both shared ends (default: 12 px).",
    )
    parser.add_argument(
        "--marking-angle-tolerance",
        type=float,
        default=3.0,
        help="Maximum undirected edge angle difference (default: 3 degrees).",
    )
    parser.add_argument(
        "--marking-min-overlap",
        type=float,
        default=30.0,
        help="Minimum shared projection length (default: 30 px).",
    )
    parser.add_argument(
        "--marking-min-overlap-ratio",
        type=float,
        default=0.6,
        help="Minimum shared / longer projected extent (default: 0.6).",
    )
    parser.add_argument(
        "--marking-max-width-ratio",
        type=float,
        default=1.5,
        help="Maximum wider / narrower shared-end width (default: 1.5).",
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="Also save topology.json and a candidate preview (opt-in).",
    )
    parser.add_argument("--topology-endpoint-tolerance", type=float, default=3.0)
    parser.add_argument(
        "--topology-max-extension",
        type=float,
        default=0.0,
        help="Maximum extension per line in pixels; 0 disables extrapolation.",
    )
    parser.add_argument("--topology-min-crossing-angle", type=float, default=15.0)
    parser.add_argument("--topology-dedup-radius", type=float, default=3.0)
    parser.add_argument(
        "--topology-min-context-length",
        type=float,
        default=30.0,
        help="Minimum unmatched-edge length for endpoint context; all are retained.",
    )
    parser.add_argument(
        "--topology-support",
        action="store_true",
        help="Add raw LSD support JSON/preview; requires --topology.",
    )
    parser.add_argument("--support-junction-radius", type=float, default=3.0)
    parser.add_argument("--support-arm-length", type=float, default=15.0)
    # Reuse public algorithm validation on empty input before accessing any files.
    args = parser.parse_args()
    if not isfinite(args.min_length) or args.min_length < 0:
        parser.error("--min-length must be finite and non-negative")
    pairing_parameters = {
        "min_width_px": args.marking_min_width,
        "max_width_px": args.marking_max_width,
        "angle_tolerance": args.marking_angle_tolerance,
        "min_overlap_px": args.marking_min_overlap,
        "min_overlap_ratio": args.marking_min_overlap_ratio,
        "max_width_ratio": args.marking_max_width_ratio,
    }
    try:
        pair_marking_edges((), **pairing_parameters)
    except ValueError as error:
        parser.error(f"Invalid marking pairing parameters: {error}")
    topology_parameters = {
        "endpoint_tolerance_px": args.topology_endpoint_tolerance,
        "max_extension_px": args.topology_max_extension,
        "min_crossing_angle": args.topology_min_crossing_angle,
        "dedup_radius_px": args.topology_dedup_radius,
        "min_context_length_px": args.topology_min_context_length,
    }
    try:
        find_topology_candidates((), **topology_parameters)
    except ValueError as error:
        parser.error(f"Invalid topology parameters: {error}")
    if args.topology_support and not args.topology:
        parser.error("--topology-support requires --topology")
    support_parameters = {
        "junction_radius_px": args.support_junction_radius,
        "arm_length_px": args.support_arm_length,
    }
    try:
        measure_topology_support(
            TopologyCandidates((), (), (), ()), (), {}, **support_parameters
        )
    except ValueError as error:
        parser.error(f"Invalid source support parameters: {error}")
    return (
        args,
        pairing_parameters,
        topology_parameters,
        (support_parameters if args.topology_support else None),
    )


def find_image_paths(samples_path: Path) -> list[Path]:
    """Resolve a single sample or naturally ordered supported images in a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if samples_path.is_file() and samples_path.suffix.lower() in extensions:
        image_paths = [samples_path]
    elif samples_path.is_dir():
        image_paths = sorted(
            (
                path
                for path in samples_path.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=image_sort_key,
        )
    else:
        raise FileNotFoundError(f"Nie znaleziono obrazu lub folderu: {samples_path}")
    if not image_paths:
        raise FileNotFoundError(f"Brak obrazów w folderze: {samples_path}")
    return image_paths


def _write_preview(path: Path, preview: NDArray[np.uint8]) -> None:
    """Save a preview and turn OpenCV's failure flag into a descriptive exception."""
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Nie udało się zapisać podglądu: {path}")


def _write_json(path: Path, data: dict[str, object], *, allow_nan: bool = True) -> None:
    """Write consistently indented diagnostic JSON with a trailing newline."""
    path.write_text(
        json.dumps(data, indent=2, allow_nan=allow_nan) + "\n", encoding="utf-8"
    )


def _unpaired_lines(
    fitted_lines: tuple[LineSegment, ...], pairs: tuple[MarkingPair, ...]
) -> tuple[LineSegment, ...]:
    """Return unmatched representatives without changing their objects or order."""
    paired_ids = {edge.segment_id for pair in pairs for edge in pair.edges}
    return tuple(line for line in fitted_lines if line.segment_id not in paired_ids)


def save_lsd_diagnostics(
    image: NDArray[np.uint8],
    segments: tuple[LineSegment, ...],
    image_path: Path,
    output_dir: Path,
    min_length: float,
) -> int:
    """Keep raw geometry intact; apply the length threshold only to its preview."""
    preview = image.copy()
    displayed = 0
    for segment in segments:
        if segment.length < min_length:
            continue
        start = (int(round(segment.start[0])), int(round(segment.start[1])))
        end = (int(round(segment.end[0])), int(round(segment.end[1])))
        cv2.line(preview, start, end, (0, 0, 255), 2)
        displayed += 1

    _write_preview(output_dir / "preview_lsd.png", preview)
    (output_dir / "source.txt").write_text(str(image_path) + "\n", encoding="utf-8")
    _write_json(
        output_dir / "segments.json",
        {
            "source": str(image_path),
            "opencv_version": cv2.__version__,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "coordinate_system": "pixels",
            "preview_min_length_px": min_length,
            "segments": [line_data(segment) for segment in segments],
        },
    )
    return displayed


def save_marking_diagnostics(
    image_path: Path,
    output_dir: Path,
    groups: tuple[SegmentGroup, ...],
    fitted_lines: tuple[LineSegment, ...],
    pairs: tuple[MarkingPair, ...],
    parameters: dict[str, float],
) -> int:
    """Save pairing provenance and return the number of unmatched representatives."""
    group_indices = {line.segment_id: i for i, line in enumerate(fitted_lines)}
    unpaired_ids = [line.segment_id for line in _unpaired_lines(fitted_lines, pairs)]
    _write_json(
        output_dir / "markings.json",
        {
            "source": str(image_path),
            "coordinate_system": "pixels",
            "interpretation": "geometric hypotheses; extents may span source gaps",
            "pairing_parameters": parameters,
            "groups": [
                {
                    "group_index": group_index,
                    "source_segment_ids": [s.segment_id for s in group.segments],
                    "fitted_line": line_data(line),
                }
                for group_index, (group, line) in enumerate(zip(groups, fitted_lines))
            ],
            "pairs": [
                {
                    "edge_segment_ids": [edge.segment_id for edge in pair.edges],
                    "group_indices": [group_indices[e.segment_id] for e in pair.edges],
                    "centerline": line_data(pair.centerline),
                    "widths_px": pair.widths_px,
                    "overlap_ratio": pair.overlap_ratio,
                }
                for pair in pairs
            ],
            "unpaired_edge_ids": unpaired_ids,
        },
    )
    return len(unpaired_ids)


def process_geometry_diagnostics(
    image: NDArray[np.uint8],
    segments: tuple[LineSegment, ...],
    image_path: Path,
    output_dir: Path,
    pairing_parameters: dict[str, float],
    topology_parameters: dict[str, float] | None,
    support_parameters: dict[str, float] | None = None,
) -> None:
    """Run the existing grouping → fitting → pairing → optional topology stages."""
    # 1. Group source fragments while retaining their separate endpoints and gaps.
    grouping_started = perf_counter()
    groups = group_segments(
        segments, angle_tolerance=3.0, distance_tolerance=2.0, gap_tolerance=12.0
    )
    grouping_seconds = perf_counter() - grouping_started
    _write_preview(output_dir / "preview_groups.png", render_groups(image, groups))

    # 2. Fit group representatives with new IDs above the entire LSD extraction.
    first_fitted_id = max((s.segment_id for s in segments), default=-1) + 1
    fitted_lines = tuple(
        fit_group_line(group, segment_id=first_fitted_id + group_index)
        for group_index, group in enumerate(groups)
    )
    fitted_path = output_dir / "preview_fitted_lines.png"
    _write_preview(fitted_path, render_fitted_lines(image, fitted_lines))

    # 3. Pair compatible edges and save the path from each axis back to its sources.
    pairing_started = perf_counter()
    pairs = pair_marking_edges(fitted_lines, **pairing_parameters)
    pairing_seconds = perf_counter() - pairing_started
    axes_path = output_dir / "preview_marking_axes.png"
    _write_preview(axes_path, render_marking_axes(image, fitted_lines, pairs))
    unpaired_count = save_marking_diagnostics(
        image_path, output_dir, groups, fitted_lines, pairs, pairing_parameters
    )

    multi_count = sum(len(group.segments) > 1 for group in groups)
    largest_group = max((len(group.segments) for group in groups), default=0)
    print(
        f"  Groups: {len(groups)} total, {multi_count} multi-segment, "
        f"{len(groups) - multi_count} single-segment, largest: {largest_group}; "
        f"grouping: {grouping_seconds:.3f} s"
    )
    print(f"  Fitted extents: {len(fitted_lines)} (may span gaps): {fitted_path.name}")
    print(
        f"  Marking hypotheses: {len(pairs)} pairs, {unpaired_count} unpaired; "
        f"pairing: {pairing_seconds:.3f} s; {axes_path.name}"
    )
    # 4. Topology is opt-in; the previous diagnostic outputs have already been saved.
    if topology_parameters is not None:
        result = save_topology_diagnostics(
            image, image_path, output_dir, fitted_lines, pairs, topology_parameters
        )
        if support_parameters is not None:
            save_topology_support_diagnostics(
                image,
                output_dir,
                result,
                pairs,
                {line.segment_id: group for line, group in zip(fitted_lines, groups)},
                support_parameters,
            )


def main() -> None:
    """Run the diagnostic pipeline for each image selected by validated CLI options."""
    # 1. Resolve configuration and sample order before creating output folders.
    args, pairing_parameters, topology_parameters, support_parameters = (
        parse_arguments()
    )
    image_paths = find_image_paths(args.samples.resolve())
    results_dir = args.output.resolve()
    detector = OpenCVSlotDetector()

    for index, image_path in enumerate(image_paths, start=1):
        loaded_image = cv2.imread(str(image_path))
        if loaded_image is None:
            print(f"Pomijam obraz, którego nie można wczytać: {image_path.name}")
            continue

        # 2. Extract raw LSD fragments from the image before deriving any geometry.
        image = cast(NDArray[np.uint8], loaded_image)
        frame = Frame(
            camera_id="test-camera",
            sequence=index - 1,
            captured_at=datetime.datetime.now(datetime.timezone.utc),
            image=image,
        )
        segments = detector.extract_segments(frame)
        output_dir = results_dir / f"image-{index}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Save source evidence, then run grouping, fitting, pairing and topology.
        displayed = save_lsd_diagnostics(
            image, segments, image_path, output_dir, args.min_length
        )
        print(
            f"[{index}/{len(image_paths)}] {image_path.name} -> {output_dir}: "
            f"{len(segments)} detected, {displayed} displayed "
            f"(>={args.min_length:g} px)",
            flush=True,
        )
        process_geometry_diagnostics(
            image,
            segments,
            image_path,
            output_dir,
            pairing_parameters,
            topology_parameters if args.topology else None,
            support_parameters,
        )
    print(f"Gotowe. Wyniki: {results_dir}")


if __name__ == "__main__":
    main()

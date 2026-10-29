import argparse
import datetime
import json
import re
from colorsys import hsv_to_rgb
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
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import group_segments


def image_sort_key(path: Path) -> list[str | int]:
    return [
        int(part) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def render_groups(
    image: NDArray[np.uint8], groups: tuple[SegmentGroup, ...]
) -> NDArray[np.uint8]:
    preview = image.copy()
    multi_count = sum(len(group.segments) > 1 for group in groups)
    colors = [
        tuple(
            round(channel * 255) for channel in hsv_to_rgb(index / multi_count, 1, 1)
        )[::-1]
        for index in range(multi_count)
    ]
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
    return {
        "segment_id": line.segment_id,
        "start": line.start,
        "end": line.end,
        "length": line.length,
    }


def main() -> None:
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

    samples_path = args.samples.resolve()
    results_dir = args.output.resolve()
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

    detector = OpenCVSlotDetector()

    for index, image_path in enumerate(image_paths, start=1):
        loaded_image = cv2.imread(str(image_path))
        if loaded_image is None:
            print(f"Pomijam obraz, którego nie można wczytać: {image_path.name}")
            continue

        image = cast(NDArray[np.uint8], loaded_image)
        frame = Frame(
            camera_id="test-camera",
            sequence=index - 1,
            captured_at=datetime.datetime.now(datetime.timezone.utc),
            image=image,
        )
        segments = detector.extract_segments(frame)

        preview = image.copy()
        displayed = 0
        for segment in segments:
            if segment.length < args.min_length:
                continue
            start = (int(round(segment.start[0])), int(round(segment.start[1])))
            end = (int(round(segment.end[0])), int(round(segment.end[1])))
            cv2.line(preview, start, end, (0, 0, 255), 2)
            displayed += 1

        output_dir = results_dir / f"image-{index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = output_dir / "preview_lsd.png"
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Nie udało się zapisać podglądu: {preview_path}")

        (output_dir / "source.txt").write_text(str(image_path) + "\n", encoding="utf-8")
        data = {
            "source": str(image_path),
            "opencv_version": cv2.__version__,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "coordinate_system": "pixels",
            "preview_min_length_px": args.min_length,
            "segments": [line_data(segment) for segment in segments],
        }
        (output_dir / "segments.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{index}/{len(image_paths)}] {image_path.name} -> {output_dir}: "
            f"{len(segments)} detected, {displayed} displayed "
            f"(>={args.min_length:g} px)",
            flush=True,
        )

        grouping_started = perf_counter()
        groups = group_segments(
            segments,
            angle_tolerance=3.0,
            distance_tolerance=2.0,
            gap_tolerance=12.0,
        )
        grouping_seconds = perf_counter() - grouping_started
        groups_path = output_dir / "preview_groups.png"
        if not cv2.imwrite(str(groups_path), render_groups(image, groups)):
            raise OSError(f"Nie udało się zapisać podglądu: {groups_path}")

        # Allocate derived IDs above the entire extraction, including other groups.
        # Representatives stay separate; never replace or regroup source fragments.
        first_fitted_id = max((s.segment_id for s in segments), default=-1) + 1
        fitted_lines = tuple(
            fit_group_line(group, segment_id=first_fitted_id + group_index)
            for group_index, group in enumerate(groups)
        )
        fitted_path = output_dir / "preview_fitted_lines.png"
        if not cv2.imwrite(str(fitted_path), render_fitted_lines(image, fitted_lines)):
            raise OSError(f"Nie udało się zapisać podglądu: {fitted_path}")

        pairing_started = perf_counter()
        pairs = pair_marking_edges(fitted_lines, **pairing_parameters)
        pairing_seconds = perf_counter() - pairing_started
        axes_path = output_dir / "preview_marking_axes.png"
        if not cv2.imwrite(
            str(axes_path), render_marking_axes(image, fitted_lines, pairs)
        ):
            raise OSError(f"Nie udało się zapisać podglądu: {axes_path}")
        group_indices = {
            line.segment_id: group_index
            for group_index, line in enumerate(fitted_lines)
        }
        paired_ids = {edge.segment_id for pair in pairs for edge in pair.edges}
        unpaired_ids = [
            line.segment_id
            for line in fitted_lines
            if line.segment_id not in paired_ids
        ]
        markings_data = {
            "source": str(image_path),
            "coordinate_system": "pixels",
            "interpretation": "geometric hypotheses; extents may span source gaps",
            "pairing_parameters": pairing_parameters,
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
        }
        (output_dir / "markings.json").write_text(
            json.dumps(markings_data, indent=2) + "\n", encoding="utf-8"
        )

        multi_count = sum(len(group.segments) > 1 for group in groups)
        single_count = len(groups) - multi_count
        largest_group = max((len(group.segments) for group in groups), default=0)
        print(
            f"  Groups: {len(groups)} total, {multi_count} multi-segment, "
            f"{single_count} single-segment, largest: {largest_group}; "
            f"grouping: {grouping_seconds:.3f} s"
        )
        print(
            f"  Fitted extents: {len(fitted_lines)} (may span gaps): {fitted_path.name}"
        )
        print(
            f"  Marking hypotheses: {len(pairs)} pairs, {len(unpaired_ids)} unpaired; "
            f"pairing: {pairing_seconds:.3f} s; {axes_path.name}"
        )

    print(f"Gotowe. Wyniki: {results_dir}")


if __name__ == "__main__":
    main()

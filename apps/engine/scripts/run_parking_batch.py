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


def main() -> None:
    parser = argparse.ArgumentParser(description="Save LSD segments and previews.")
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
    args = parser.parse_args()
    if not isfinite(args.min_length) or args.min_length < 0:
        parser.error("--min-length must be finite and non-negative")

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
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "length": segment.length,
                }
                for segment in segments
            ],
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

        multi_count = sum(len(group.segments) > 1 for group in groups)
        single_count = len(groups) - multi_count
        largest_group = max((len(group.segments) for group in groups), default=0)
        print(
            f"  Groups: {len(groups)} total, {multi_count} multi-segment, "
            f"{single_count} single-segment, largest: {largest_group}; "
            f"grouping: {grouping_seconds:.3f} s"
        )

    print(f"Gotowe. Wyniki: {results_dir}")


if __name__ == "__main__":
    main()

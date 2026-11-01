"""Offline preview of saved camera ROIs; no slot detection or occupancy inference."""

import argparse
import json
from colorsys import hsv_to_rgb
from hashlib import sha256
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .configuration import ParkingConfiguration, load_configuration
from .domain.types import CameraId, ParkingSlotId

_PANEL_WIDTH = 440
_PADDING = 16
_FONT_SIZE = 20


def _slot_color(slot_id: ParkingSlotId) -> tuple[int, int, int]:
    # Unlike Python's hash(), this keeps the same slot color across processes/views.
    hue = int.from_bytes(sha256(slot_id.encode("utf-8")).digest()[:2], "big") / 65536
    red, green, blue = hsv_to_rgb(hue, 0.45, 1.0)
    return round(blue * 255), round(green * 255), round(red * 255)


def _quoted(value: str) -> str:
    # Preserve Unicode; make embedded newlines/control characters explicit in labels.
    return json.dumps(value, ensure_ascii=False)


def render_configuration_preview(
    configuration: ParkingConfiguration,
    camera_id: CameraId,
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Draw saved projections on a copy, with full IDs/labels in a side legend.

    The original image keeps its scale and origin at (0, 0). The PNG canvas grows
    to the right and, if the legend needs it, downward. Colors identify slots,
    not occupancy. Numbered badges only refer to the accompanying legend.
    """
    if image.dtype != np.uint8:
        raise TypeError("Expected a uint8 BGR image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3)")
    height, width = image.shape[:2]
    projections = configuration.projections_for_camera(
        camera_id, image_width=width, image_height=height
    )
    camera = next(camera for camera in configuration.cameras if camera.id == camera_id)
    slots = {slot.id: slot for slot in configuration.slots}
    numbers = {slot_id: index for index, slot_id in enumerate(sorted(slots), start=1)}

    # OpenCV 5's bundled sans font supports Unicode, including Polish labels.
    # Legacy Hershey putText would replace those characters with question marks.
    font = cv2.FontFace()
    if not font.set("sans"):
        raise ValueError("OpenCV's bundled sans font is unavailable")
    white = (235, 235, 235)
    rows: list[tuple[str, tuple[int, int, int]]] = [
        (
            (
                "Saved parking geometry"
                if configuration.is_approved
                else "DRAFT - NOT APPROVED"
            ),
            white,
        ),
        (f"parking: {_quoted(configuration.parking.id)}", white),
        (f"name: {_quoted(configuration.parking.name)}", white),
        (f"revision: {configuration.revision}", white),
        (f"camera: {_quoted(camera.id)}", white),
        (f"camera name: {_quoted(camera.name)}", white),
        (f"reference: {width} x {height} px", white),
        (f"configured projections: {len(projections)}", white),
        ("Colors identify slots, not occupancy.", white),
    ]
    for projection in projections:
        slot = slots[projection.parking_slot_id]
        color = _slot_color(slot.id)
        rows.append((f"[{numbers[slot.id]}] id: {_quoted(slot.id)}", color))
        label = "(none)" if slot.label is None else _quoted(slot.label)
        rows.append((f"label: {label}", white))
    if not projections:
        rows.append(("No configured ROI for this camera.", white))

    row_heights = []
    for text, _ in rows:
        _, _, _, text_height = cv2.getTextSize(
            (_PANEL_WIDTH, 0),
            text,
            (_PADDING, _FONT_SIZE),
            font,
            _FONT_SIZE,
            flags=cv2.PUT_TEXT_WRAP,
            wrap=(_PADDING, _PANEL_WIDTH - _PADDING),
        )
        row_heights.append(max(_FONT_SIZE, text_height) + 12)
    canvas_height = max(height, sum(row_heights) + 2 * _PADDING)
    preview = np.full((canvas_height, width + _PANEL_WIDTH, 3), 28, dtype=np.uint8)
    preview[:height, :width] = image
    # A view restricted to the reference region prevents drawing into the legend.
    reference = preview[:height, :width]
    for projection in projections:
        color = _slot_color(projection.parking_slot_id)
        # Schema v1 uses centers of extreme pixels. Retain subpixel geometry.
        points = np.array(
            [
                (round(p.x * (width - 1) * 256), round(p.y * (height - 1) * 256))
                for p in projection.polygon.vertices
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        for stroke, thickness in (((0, 0, 0), 5), (color, 2)):
            cv2.polylines(
                reference, [points], True, stroke, thickness, cv2.LINE_AA, shift=8
            )
        text = str(numbers[projection.parking_slot_id])
        _, _, text_width, _ = cv2.getTextSize((width, height), text, (0, 0), font, 18)
        badge_width, badge_height = text_width + 10, 28
        if width >= badge_width and height >= badge_height:
            first = projection.polygon.vertices[0]
            x = min(round(first.x * (width - 1)) + 6, width - badge_width)
            y = min(round(first.y * (height - 1)) + 6, height - badge_height)
            cv2.rectangle(
                reference, (x, y), (x + badge_width - 1, y + badge_height - 1), 0, -1
            )
            cv2.putText(reference, text, (x + 5, y + 21), color, font, 18)

    if not configuration.is_approved:
        cv2.rectangle(reference, (8, 8), (min(width - 1, 340), 48), (0, 0, 0), -1)
        cv2.putText(
            reference, "DRAFT - NOT APPROVED", (16, 36), (0, 200, 255), font, 24
        )

    y = _PADDING + _FONT_SIZE
    for (text, color), row_height in zip(rows, row_heights):
        cv2.putText(
            preview,
            text,
            (width + _PADDING, y),
            color,
            font,
            _FONT_SIZE,
            flags=cv2.PUT_TEXT_WRAP,
            wrap=(width + _PADDING, width + _PANEL_WIDTH - _PADDING),
        )
        y += row_height
    return preview


def save_configuration_preview(
    configuration_path: Path, image_path: Path, camera_id: CameraId, output_path: Path
) -> None:
    """Read both inputs, validate and encode, then exclusively create a new PNG."""
    save_geometry_preview(
        load_configuration(configuration_path), image_path, camera_id, output_path
    )


def save_geometry_preview(
    configuration: ParkingConfiguration,
    image_path: Path,
    camera_id: CameraId,
    output_path: Path,
) -> None:
    """Preview validated domain objects, including explicitly marked drafts."""
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output must have a .png extension")
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {image_path}")
    # EXIF autorotation would change the configured coordinate system.
    image = cv2.imread(
        str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise ValueError(f"Cannot decode reference image: {image_path}")
    preview = render_configuration_preview(
        configuration, camera_id, cast(NDArray[np.uint8], image)
    )
    success, encoded = cv2.imencode(".png", preview)
    if not success:
        raise OSError("Cannot encode preview as PNG")
    # Also protects input paths and aliases/symlinks: an existing file is never opened
    # for writing, even when it is the requested output path.
    with output_path.open("xb") as output:
        output.write(encoded.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview saved parking camera ROIs")
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--camera-id", type=CameraId, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        save_configuration_preview(
            args.configuration, args.image, args.camera_id, args.output
        )
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(2, f"Cannot create configuration preview: {error}\n")
    print(f"Saved configuration preview: {args.output}")


if __name__ == "__main__":
    main()

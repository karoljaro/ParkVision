"""Import whole-slot LabelMe polygons as DRAFT; approve only as an explicit step."""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

import cv2

from .configuration import (
    ParkingConfiguration,
    _array,
    _integer,
    _record,
    _string,
    _unique_object,
    save_configuration,
)
from .configuration_preview import save_geometry_preview
from .domain.camera import Camera
from .domain.geometry.point import Point
from .domain.geometry.polygon import Polygon
from .domain.parking import Parking
from .domain.parking_slot import ParkingSlot
from .domain.slot_projection import SlotProjection
from .domain.types import CameraId, ParkingId, ParkingSlotId


def _pixel(value: object, extent: int) -> float:
    if type(value) not in (int, float):
        raise ValueError("LabelMe coordinates must be numbers, not bool/strings")
    try:
        return float(cast(int | float, value)) / (extent - 1)
    except OverflowError as error:
        raise ValueError("LabelMe coordinate is too large") from error


def load_labelme_configuration(
    annotation_path: Path,
    image_path: Path,
    *,
    reviewed_annotation_sha256: str | None = None,
) -> ParkingConfiguration:
    """Read one camera's slot annotations; always return unapproved geometry.

    LabelMe label = persistent slot ID; description = optional display label.
    The parkvision extension binds parking/camera IDs and the exact reference.
    Existing domain constructors validate normalized geometry and relationships.
    """
    raw = annotation_path.read_bytes()
    if reviewed_annotation_sha256 is not None:
        if sha256(raw).hexdigest() != reviewed_annotation_sha256:
            raise ValueError("Annotation differs from the reviewed SHA256")
    value: object = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("LabelMe annotation must be an object")
    data = cast(dict[str, object], value)
    metadata = _record(
        data.get("parkvision"),
        "task parking_id parking_name camera_id camera_name revision image_sha256",
        "parkvision metadata",
    )
    if metadata["task"] != "parking-slot-setup":
        raise ValueError(
            "Expected whole-slot setup, not paint segmentation annotations"
        )
    expected_hash = _string(metadata["image_sha256"], "image_sha256")
    if len(expected_hash) != 64 or any(
        c not in "0123456789abcdef" for c in expected_hash
    ):
        raise ValueError("image_sha256 must be a lowercase SHA256 digest")
    reference = _string(data.get("imagePath"), "imagePath")
    if (annotation_path.parent / reference).resolve() != image_path.resolve():
        raise ValueError("LabelMe imagePath references a different image")
    if data.get("imageData") is not None:
        raise ValueError("Use LabelMe --nodata; embedded imageData is not accepted")
    image_bytes = image_path.read_bytes()
    if sha256(image_bytes).hexdigest() != expected_hash:
        raise ValueError("Reference image differs from image_sha256")
    image = cv2.imread(
        str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise ValueError("Cannot decode LabelMe reference image")
    height, width = image.shape[:2]
    if (
        _integer(data.get("imageWidth"), "imageWidth", minimum=2),
        _integer(data.get("imageHeight"), "imageHeight", minimum=2),
    ) != (width, height):
        raise ValueError("LabelMe dimensions differ from reference image")
    parking_id = ParkingId(_string(metadata["parking_id"], "parking_id"))
    camera_id = CameraId(_string(metadata["camera_id"], "camera_id"))
    parking = Parking(parking_id, _string(metadata["parking_name"], "parking_name"))
    camera = Camera(
        camera_id, parking_id, _string(metadata["camera_name"], "camera_name")
    )
    slots = []
    projections = []
    for value in _array(data.get("shapes"), "shapes"):
        if not isinstance(value, dict):
            raise ValueError("Every LabelMe shape must be an object")
        shape = cast(dict[str, object], value)
        if any(key in shape for key in ("parking_id", "camera_id", "slot_id")):
            raise ValueError(
                "Use parkvision metadata for relationships and label for slot_id"
            )
        if shape.get("shape_type") != "polygon" or shape.get("group_id") is not None:
            raise ValueError("Use one polygon per slot, without group_id")
        if shape.get("mask") is not None:
            raise ValueError("Use polygon vertices, not an embedded shape mask")
        slot_id = ParkingSlotId(_string(shape.get("label"), "shape.label (slot_id)"))
        description = shape.get("description")
        label = (
            None if description in (None, "") else _string(description, "description")
        )
        vertices = []
        for value in _array(shape.get("points"), "points"):
            pair = _array(value, "point")
            if len(pair) != 2:
                raise ValueError("LabelMe points must contain x and y")
            vertices.append(Point(_pixel(pair[0], width), _pixel(pair[1], height)))
        slots.append(ParkingSlot(slot_id, parking_id, label))
        projections.append(SlotProjection(camera_id, slot_id, Polygon(tuple(vertices))))
    return ParkingConfiguration(
        parking=parking,
        cameras=(camera,),
        slots=tuple(slots),
        projections=tuple(projections),
        camera_image_sizes={camera_id: (width, height)},
        revision=_integer(metadata["revision"], "revision", minimum=1),
    )


def approve_labelme_configuration(
    annotation_path: Path,
    image_path: Path,
    output_path: Path,
    *,
    reviewed_annotation_sha256: str,
    approved_by: str,
    approved_at: datetime,
) -> None:
    """Record caller-supplied approval for exact reviewed bytes, without defaults.

    This is an operator declaration, not proof of identity or of visual review.
    The caller must obtain actual approval before invoking this operation.
    """
    draft = load_labelme_configuration(
        annotation_path,
        image_path,
        reviewed_annotation_sha256=reviewed_annotation_sha256,
    )
    approved = replace(draft, approved_by=approved_by, approved_at=approved_at)
    save_configuration(approved, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser(
        "preview", help="Validate LabelMe and render DRAFT PNG"
    )
    approve = commands.add_parser(
        "approve", help="Export after actual operator approval"
    )
    for command in (preview, approve):
        command.add_argument("--annotation", type=Path, required=True)
        command.add_argument("--image", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    approve.add_argument("--reviewed-annotation-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True, help="ISO 8601 with timezone")
    args = parser.parse_args()
    try:
        if args.command == "preview":
            digest = sha256(args.annotation.read_bytes()).hexdigest()
            draft = load_labelme_configuration(
                args.annotation, args.image, reviewed_annotation_sha256=digest
            )
            save_geometry_preview(draft, args.image, draft.cameras[0].id, args.output)
            print("DRAFT - NOT APPROVED")
            print(f"Annotation SHA256: {digest}")
        else:
            approve_labelme_configuration(
                args.annotation,
                args.image,
                args.output,
                reviewed_annotation_sha256=args.reviewed_annotation_sha256,
                approved_by=args.approved_by,
                approved_at=datetime.fromisoformat(args.approved_at),
            )
            print(f"Saved approved configuration: {args.output}")
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(2, f"Cannot import LabelMe slot setup: {error}\n")


if __name__ == "__main__":
    main()

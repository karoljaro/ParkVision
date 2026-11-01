"""Versioned, operator-approved geometry, independent of frames and detectors."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import cast

from .domain.camera import Camera
from .domain.geometry.point import Point
from .domain.geometry.polygon import Polygon
from .domain.parking import Parking
from .domain.parking_slot import ParkingSlot
from .domain.slot_projection import SlotProjection
from .domain.types import CameraId, ParkingId, ParkingSlotId

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ParkingConfiguration:
    """One approved revision; existing domain objects remain the geometry model.

    A projection means an approved, usable ROI for that camera/slot pair.
    Absence means no configured ROI, not a current occupancy/occlusion state.
    Approval fields record an operator declaration, not an authenticated signature.
    """

    parking: Parking
    cameras: tuple[Camera, ...]
    slots: tuple[ParkingSlot, ...]
    projections: tuple[SlotProjection, ...]
    camera_image_sizes: Mapping[CameraId, tuple[int, int]]
    revision: int
    approved_by: str
    approved_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", tuple(self.cameras))
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "projections", tuple(self.projections))
        sizes = {key: tuple(size) for key, size in self.camera_image_sizes.items()}
        object.__setattr__(self, "camera_image_sizes", MappingProxyType(sizes))
        _integer(self.revision, "revision", minimum=1)
        _string(self.approved_by, "approved_by")
        if self.approved_at.utcoffset() is None:
            raise ValueError("approved_at must include a timezone")
        _string(self.parking.id, "parking.id")
        _string(self.parking.name, "parking.name")
        camera_ids: set[CameraId] = set()
        for camera in self.cameras:
            _string(camera.id, "camera.id")
            _string(camera.name, "camera.name")
            if camera.id in camera_ids:
                raise ValueError(f"Duplicate camera ID: {camera.id}")
            if camera.parking_id != self.parking.id:
                raise ValueError(f"Camera {camera.id} belongs to another parking")
            camera_ids.add(camera.id)
        if not camera_ids or set(sizes) != camera_ids:
            raise ValueError("Image sizes must cover exactly the configured cameras")
        for size in sizes.values():
            if len(size) != 2:
                raise ValueError("Image size must contain width and height")
            _integer(size[0], "image_width", minimum=2)
            _integer(size[1], "image_height", minimum=2)
        slot_ids: set[ParkingSlotId] = set()
        for slot in self.slots:
            _string(slot.id, "slot.id")
            if slot.label is not None:
                _string(slot.label, "slot.label")
            if slot.id in slot_ids:
                raise ValueError(f"Duplicate slot ID: {slot.id}")
            if slot.parking_id != self.parking.id:
                raise ValueError(f"Slot {slot.id} belongs to another parking")
            slot_ids.add(slot.id)
        pairs: set[tuple[CameraId, ParkingSlotId]] = set()
        for projection in self.projections:
            pair = (projection.camera_id, projection.parking_slot_id)
            if pair[0] not in camera_ids or pair[1] not in slot_ids:
                raise ValueError(
                    f"Projection references an unknown camera/slot: {pair}"
                )
            if pair in pairs:
                raise ValueError(f"Duplicate camera/slot projection: {pair}")
            pairs.add(pair)
            for vertex in projection.polygon.vertices:
                _coordinate(vertex.x)
                _coordinate(vertex.y)
        if not slot_ids or {slot_id for _, slot_id in pairs} != slot_ids:
            raise ValueError("Every configured slot needs at least one projection")

    def projections_for_camera(
        self, camera_id: CameraId, *, image_width: int, image_height: int
    ) -> tuple[SlotProjection, ...]:
        """Return saved ROIs after checking the full-frame reference dimensions.

        Matching dimensions cannot prove that camera pose/crop/lens stayed fixed.
        Those changes require an explicit operator review and a new revision.
        """
        _integer(image_width, "image_width", minimum=2)
        _integer(image_height, "image_height", minimum=2)
        if camera_id not in self.camera_image_sizes:
            raise ValueError(f"Unknown camera: {camera_id}")
        if self.camera_image_sizes[camera_id] != (image_width, image_height):
            raise ValueError(f"Image dimensions differ from setup for {camera_id}")
        return tuple(p for p in self.projections if p.camera_id == camera_id)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without edge whitespace")
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _coordinate(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Coordinates must be finite numbers in [0, 1]")
    try:
        number = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError("Coordinate is too large") from error
    if not isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Coordinates must be finite numbers in [0, 1]")
    return number


def _record(value: object, keys: str, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"{field} must be an object with exactly: {keys}")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_configuration(path: Path) -> ParkingConfiguration:
    """Load schema v1 strictly; never infer, repair, approve or migrate geometry."""
    data = _record(
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object),
        "schema_version revision approved_by approved_at "
        "parking cameras slots projections",
        "configuration",
    )
    version = _integer(data["schema_version"], "schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {version}")
    parking_data = _record(data["parking"], "id name", "parking")
    parking = Parking(
        ParkingId(_string(parking_data["id"], "parking.id")),
        _string(parking_data["name"], "parking.name"),
    )
    cameras = []
    sizes: dict[CameraId, tuple[int, int]] = {}
    for value in _array(data["cameras"], "cameras"):
        item = _record(value, "id parking_id name image_width image_height", "camera")
        camera_id = CameraId(_string(item["id"], "camera.id"))
        cameras.append(
            Camera(
                camera_id,
                ParkingId(_string(item["parking_id"], "camera.parking_id")),
                _string(item["name"], "camera.name"),
            )
        )
        sizes[camera_id] = (
            _integer(item["image_width"], "image_width", minimum=2),
            _integer(item["image_height"], "image_height", minimum=2),
        )
    slots = []
    for value in _array(data["slots"], "slots"):
        item = _record(value, "id parking_id label", "slot")
        slots.append(
            ParkingSlot(
                ParkingSlotId(_string(item["id"], "slot.id")),
                ParkingId(_string(item["parking_id"], "slot.parking_id")),
                None if item["label"] is None else _string(item["label"], "slot.label"),
            )
        )
    projections = []
    for value in _array(data["projections"], "projections"):
        item = _record(value, "camera_id parking_slot_id polygon", "projection")
        vertices = []
        for vertex in _array(item["polygon"], "polygon"):
            point = _record(vertex, "x y", "vertex")
            vertices.append(Point(_coordinate(point["x"]), _coordinate(point["y"])))
        projections.append(
            SlotProjection(
                CameraId(_string(item["camera_id"], "projection.camera_id")),
                ParkingSlotId(_string(item["parking_slot_id"], "parking_slot_id")),
                Polygon(tuple(vertices)),
            )
        )
    return ParkingConfiguration(
        parking=parking,
        cameras=tuple(cameras),
        slots=tuple(slots),
        projections=tuple(projections),
        camera_image_sizes=sizes,
        revision=_integer(data["revision"], "revision", minimum=1),
        approved_by=_string(data["approved_by"], "approved_by"),
        approved_at=datetime.fromisoformat(_string(data["approved_at"], "approved_at")),
    )


def save_configuration(configuration: ParkingConfiguration, path: Path) -> None:
    """Export to a new file only; revisions are never overwritten implicitly."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "revision": configuration.revision,
        "approved_by": configuration.approved_by,
        "approved_at": configuration.approved_at.isoformat(),
        "parking": {"id": configuration.parking.id, "name": configuration.parking.name},
        "cameras": [
            {
                "id": camera.id,
                "parking_id": camera.parking_id,
                "name": camera.name,
                "image_width": configuration.camera_image_sizes[camera.id][0],
                "image_height": configuration.camera_image_sizes[camera.id][1],
            }
            for camera in configuration.cameras
        ],
        "slots": [
            {"id": slot.id, "parking_id": slot.parking_id, "label": slot.label}
            for slot in configuration.slots
        ],
        "projections": [
            {
                "camera_id": projection.camera_id,
                "parking_slot_id": projection.parking_slot_id,
                "polygon": [
                    {"x": vertex.x, "y": vertex.y}
                    for vertex in projection.polygon.vertices
                ],
            }
            for projection in configuration.projections
        ],
    }
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved parking geometry")
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    try:
        configuration = load_configuration(args.configuration)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Invalid configuration: {error}\n")
    print(
        f"Valid configuration: {configuration.parking.id}, "
        f"revision {configuration.revision}, "
        f"{len(configuration.cameras)} cameras, "
        f"{len(configuration.slots)} slots, "
        f"{len(configuration.projections)} projections; "
        f"approval recorded by {configuration.approved_by}"
    )


if __name__ == "__main__":
    main()

"""Check source joins and propose image-local review pairs without labeling them."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Any

from research.datasets_v4.contracts import Asset, Fact, Role

from .source import Bundle, FrameSource

Point = tuple[float, float]
Polygon = tuple[Point, ...]
Box = tuple[float, float, float, float]


class Relation(StrEnum):
    OWNING = "owning_slot"
    PERSPECTIVE = "perspective_overlap_only"
    BLOCKS = "physically_blocks_neighbor"
    UNCERTAIN = "uncertain_or_occluded"
    NOT_ASSESSED = "not_assessed"


class Link(StrEnum):
    MATCHED = "unique_source_id_match"
    NO_VEHICLE = "source_minus_one"
    OTHER_OBJECT = "source_minus_two"
    INVALID_REFERENCE = "invalid_car_reference"
    MISSING_MASK = "missing_mask"
    INVALID_MASK = "invalid_mask"
    AMBIGUOUS = "ambiguous_car_id"
    METADATA_CONFLICT = "conflicting_image_metadata"


@dataclass(frozen=True, slots=True)
class Shape:
    polygons: tuple[Polygon, ...]
    bounds: Box

    def inside(self, rect: Box | None) -> bool | None:
        if rect is None:
            return None
        x, y, width, height = rect
        a, b, c, d = self.bounds
        return x <= a and y <= b and c < x + width and d < y + height


@dataclass(frozen=True, slots=True)
class Slot:
    annotation_id: int
    camera_slot_key: str | None
    car_id: int | None
    category_id: int
    link: Link
    car_annotation_ids: tuple[int, ...]
    inside_annotation_rectangle: bool | None
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Vehicle:
    annotation_id: int
    car_id: int | None
    inside_annotation_rectangle: bool | None
    mask_valid: bool
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pair:
    image_id: int
    car_annotation_id: int
    spot_annotation_id: int
    reason: str
    relation: Relation = Relation.NOT_ASSESSED
    reviewer: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Frame:
    image_id: int
    image: Asset
    original_filename: str
    width: int
    height: int
    registry_record: str
    historical_role: Role
    v4_effective_role: Role
    review_role: Role
    site: Fact
    camera: Fact
    session: Fact
    day: Fact
    source_capture_time: str
    source_subset: Asset
    cars_frame_present: bool
    annotation_rectangle: tuple[int, ...] | None
    issues: tuple[str, ...]
    slots: tuple[Slot, ...]
    vehicles: tuple[Vehicle, ...]
    pairs: tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class Queue:
    registry_sha256: str
    policy_sha256: str
    v4_manifest_sha256: str
    diagnostic_manifest_sha256: str
    bundles: tuple[Bundle, ...]
    frames: tuple[Frame, ...]
    parent_queue_sha256: str | None = None
    training_eligible: bool = False
    schema_version: int = 1


def polygons(annotation: dict[str, Any], size: tuple[int, int]) -> Shape:
    data = annotation.get("segmentation")
    if not isinstance(data, list) or not data:
        raise ValueError("missing or unsupported polygon segmentation")
    result = []
    width, height = size
    for flat in data:
        if (
            not isinstance(flat, list)
            or len(flat) < 6
            or len(flat) % 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in flat)
        ):
            raise ValueError("malformed polygon coordinates")
        points = tuple(zip(flat[::2], flat[1::2], strict=True))
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError("polygon coordinates outside source image")
        twice_area = sum(
            x * v - u * y
            for (x, y), (u, v) in zip(points, points[1:] + points[:1], strict=True)
        )
        if not twice_area:
            raise ValueError("degenerate polygon")
        result.append(points)
    xs, ys = zip(*(point for shape in result for point in shape), strict=True)
    return Shape(tuple(result), (min(xs), min(ys), max(xs), max(ys)))


def rectangle(image: dict[str, Any]) -> Box | None:
    rect = image.get("annotationsRectangle")
    if rect is None:
        return None
    if (
        not isinstance(rect, list)
        or len(rect) != 4
        or any(type(value) is not int for value in rect)
    ):
        raise ValueError("invalid annotationsRectangle")
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) <= 0:
        raise ValueError("invalid annotationsRectangle")
    if x + w > image["width"] or y + h > image["height"]:
        raise ValueError("annotationsRectangle outside image")
    return x, y, w, h


def camera_slot_key(camera: str, shape: Shape) -> str:
    # Canonical order for the same polygon, including reversed winding.
    canonical = []
    for points in shape.polygons:
        if points[0] == points[-1]:
            points = points[:-1]
        rotations = []
        for direction in (points, points[::-1]):
            rotations.extend(direction[i:] + direction[:i] for i in range(len(points)))
        canonical.append(min(rotations))
    encoded = json.dumps(sorted(canonical), separators=(",", ":")).encode()
    return f"{camera}:slot-geometry:{hashlib.sha256(encoded).hexdigest()[:20]}"


def read_shape(
    row: dict[str, Any], size: tuple[int, int]
) -> tuple[Shape | None, list[str]]:
    issues = []
    try:
        shape = polygons(row, size)
    except ValueError as exc:
        return None, [str(exc)]
    bbox = row.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
        or min(bbox[2:]) <= 0
    ):
        issues.append("invalid source bbox")
    else:
        x, y, w, h = bbox
        if any(
            abs(a - b) > 2
            for a, b in zip(shape.bounds, (x, y, x + w, y + h), strict=True)
        ):
            issues.append("source bbox differs from polygon bounds by over two pixels")
    return shape, issues


def join(
    spots: FrameSource, cars: FrameSource | None, camera: str
) -> tuple[tuple[Slot, ...], tuple[Vehicle, ...], tuple[Pair, ...], tuple[str, ...]]:
    image = spots.image
    for frame in (spots, cars):
        if frame is not None and any(
            row.get("image_id") != frame.image["id"] for row in frame.annotations
        ):
            raise ValueError("Annotation refers to a different image")
    size = (image["width"], image["height"])
    rect = rectangle(image)
    issues = []
    cars_rect = rectangle(cars.image) if cars else None
    metadata_ok = cars is not None and all(
        image.get(key) == cars.image.get(key)
        for key in (
            "id",
            "file_name",
            "width",
            "height",
            "date",
            "time",
            "dataset",
            "subset",
            "climate",
            "annotationsRectangle",
        )
    )
    if cars is None:
        issues.append("cars_frame_missing")
    elif not metadata_ok:
        issues.append("cars_spots_image_metadata_conflict")
    if rect is None:
        issues.append("annotations_rectangle_unknown")
    indexed = defaultdict(list)
    vehicle_shapes = {}
    vehicles = []
    all_cars = cars.annotations if cars else []
    counts = Counter(row["id"] for row in all_cars)
    for row in all_cars:
        identity = row["id"]
        car_id = row.get("car_id")
        if type(car_id) is int and car_id >= 0:
            indexed[car_id].append(identity)
        shape, warnings = read_shape(row, size)
        if counts[identity] > 1:
            warnings.append("duplicate car annotation id")
            shape = None
        if row.get("category_id") != 1 or row.get("iscrowd") != 0:
            warnings.append("unexpected vehicle category or crowd annotation")
            shape = None
        if type(car_id) is not int or car_id < 0:
            warnings.append("invalid vehicle car_id")
            car_id = None
        if shape:
            vehicle_shapes[identity] = shape
        vehicles.append(
            Vehicle(
                identity,
                car_id,
                shape.inside(cars_rect) if shape else None,
                shape is not None,
                tuple(warnings),
            )
        )
    spot_shapes = {}
    slots = []
    keys = Counter(row["id"] for row in spots.annotations)
    if any(count != 1 for count in keys.values()):
        raise ValueError(f"Duplicate spot annotation ID in frame {image['id']}")
    for row in spots.annotations:
        shape, warnings = read_shape(row, size)
        car_id = row.get("car_id")
        choices = ()
        category = row["category_id"]
        if type(car_id) is not int or car_id < -2:
            status = Link.INVALID_REFERENCE
            car_id = None if type(car_id) is not int else car_id
        elif (category == 0 and car_id != -1) or (
            category != 0 and (category != 1 or car_id == -1)
        ):
            status = Link.INVALID_REFERENCE
        elif car_id == -1:
            status = Link.NO_VEHICLE
        elif car_id == -2:
            status = Link.OTHER_OBJECT
        elif cars is not None and not metadata_ok:
            status = Link.METADATA_CONFLICT
        else:
            choices = tuple(sorted(indexed[car_id]))
            if not choices:
                status = Link.MISSING_MASK
            elif len(choices) != 1:
                status = Link.AMBIGUOUS
            elif choices[0] not in vehicle_shapes:
                status = Link.INVALID_MASK
            else:
                status = Link.MATCHED
        if shape:
            spot_shapes[row["id"]] = shape
        slots.append(
            Slot(
                row["id"],
                camera_slot_key(camera, shape) if shape else None,
                car_id,
                category,
                status,
                choices,
                shape.inside(rect) if shape else None,
                tuple(warnings),
            )
        )
    pairs = []
    if metadata_ok:
        for slot in slots:
            shape = spot_shapes.get(slot.annotation_id)
            if shape is None:
                continue
            x, y, right, bottom = shape.bounds
            for identity, vehicle in sorted(vehicle_shapes.items()):
                a, b, c, d = vehicle.bounds
                # Proximity selects work for the operator, never a target class.
                nearby = (
                    a <= right + 12 and c >= x - 12 and b <= bottom + 12 and d >= y - 12
                )
                if identity in slot.car_annotation_ids or nearby:
                    reason = (
                        "source_car_id"
                        if identity in slot.car_annotation_ids
                        else "near_slot"
                    )
                    pairs.append(
                        Pair(image["id"], identity, slot.annotation_id, reason)
                    )
    return tuple(slots), tuple(vehicles), tuple(pairs), tuple(issues)

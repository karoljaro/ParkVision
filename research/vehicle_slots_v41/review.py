"""Apply explicit operator decisions to a separate, diagnostic-only revision."""

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from research.datasets_v4.contracts import Role
from research.datasets_v4.storage import fingerprint

from .relations import Pair, Queue, Relation


@dataclass(frozen=True, slots=True)
class Review:
    image_id: int
    car_annotation_id: int
    spot_annotation_id: int
    relation: Relation
    reviewer: str
    note: str

    def __post_init__(self):
        if (
            self.relation == Relation.NOT_ASSESSED
            or not self.reviewer.strip()
            or not self.note.strip()
        ):
            raise ValueError("A manual decision requires a reviewer and a note")


def read_csv(path: Path) -> tuple[Review, ...]:
    reviews = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        required = {
            "image_id",
            "car_annotation_id",
            "spot_annotation_id",
            "relation",
            "reviewer",
            "note",
        }
        if not rows.fieldnames or not required <= set(rows.fieldnames):
            raise ValueError("Missing columns in review CSV")
        for row in rows:
            relation = Relation(row["relation"])
            if relation != Relation.NOT_ASSESSED:
                reviews.append(
                    Review(
                        int(row["image_id"]),
                        int(row["car_annotation_id"]),
                        int(row["spot_annotation_id"]),
                        relation,
                        row["reviewer"],
                        row["note"],
                    )
                )
    return tuple(reviews)


def apply(queue: Queue, reviews: tuple[Review, ...]) -> Queue:
    if queue.schema_version != 1 or queue.training_eligible:
        raise ValueError("Expected a diagnostic v4.1 queue")
    grouped: dict[int, list[Review]] = {}
    seen = set()
    for review in reviews:
        key = (review.image_id, review.car_annotation_id, review.spot_annotation_id)
        if key in seen:
            raise ValueError(f"Repeated operator decision for {key}")
        seen.add(key)
        grouped.setdefault(review.image_id, []).append(review)
    frames = []
    for frame in queue.frames:
        if frame.review_role != Role.DIAGNOSTIC:
            raise ValueError("A/B queue frame must remain diagnostic")
        vehicles = {car.annotation_id for car in frame.vehicles if car.mask_valid}
        slots = {slot.annotation_id for slot in frame.slots if slot.camera_slot_key}
        pairs = {(p.car_annotation_id, p.spot_annotation_id): p for p in frame.pairs}
        for row in grouped.pop(frame.image_id, []):
            key = (row.car_annotation_id, row.spot_annotation_id)
            if key[0] not in vehicles or key[1] not in slots:
                raise ValueError(
                    f"Review references unavailable source geometry: {key}"
                )
            if "cars_spots_image_metadata_conflict" in frame.issues:
                raise ValueError("Cannot assess pairs with conflicting image metadata")
            previous = pairs.get(key, Pair(frame.image_id, *key, "operator_added"))
            pairs[key] = replace(
                previous,
                relation=row.relation,
                reviewer=row.reviewer.strip(),
                note=row.note.strip(),
            )
        frames.append(replace(frame, pairs=tuple(pairs.values())))
    if grouped:
        raise ValueError(f"Review references unknown frame: {next(iter(grouped))}")
    return replace(queue, frames=tuple(frames), parent_queue_sha256=fingerprint(queue))

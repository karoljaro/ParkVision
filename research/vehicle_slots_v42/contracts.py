"""Separate visual suggestions, source evidence and operator decisions."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from research.datasets_v4.contracts import Asset, Role
from research.vehicle_slots_v41.relations import Relation
from research.vehicle_slots_v41.source import Bundle


class Status(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class Geometry(StrEnum):
    SUPPORTED = "supported_by_visible_boundaries"
    PARTIAL = "partially_supported"
    UNVERIFIABLE = "unverifiable"


class Visibility(StrEnum):
    CLEAR = "clear"
    PARTIAL = "partial"
    OBSCURED = "obscured"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class PairKey:
    image_id: int
    car_annotation_id: int
    spot_annotation_id: int

    def __post_init__(self) -> None:
        if min(self.image_id, self.car_annotation_id, self.spot_annotation_id) <= 0:
            raise ValueError("Pair IDs must be positive")

    @property
    def text(self) -> str:
        return f"{self.image_id}:{self.car_annotation_id}:{self.spot_annotation_id}"


@dataclass(frozen=True, slots=True)
class Suggestion:
    suggested_relation: Relation
    geometry: Geometry
    visibility: Visibility
    confidence: Confidence
    relation_note: str
    geometry_note: str

    def __post_init__(self) -> None:
        if not self.relation_note.strip() or not self.geometry_note.strip():
            raise ValueError(
                "A visual suggestion needs separate evidence for both checks"
            )


@dataclass(frozen=True, slots=True)
class Selection:
    pair: PairKey
    suggestion: Suggestion


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    pair: PairKey
    car_id: int
    camera_slot_key: str
    site_id: str
    camera_id: str
    session_id: str | None
    capture_time: str
    registry_record: str
    source: Asset
    source_subset: Asset
    historical_role: Role
    source_owners: tuple[int, ...]
    neighbors: tuple[int, ...]
    original_view: str
    full_overlay: str
    detail_original: str
    detail_overlay: str
    suggestion: Suggestion


@dataclass(frozen=True, slots=True)
class Batch:
    queue: Asset
    queue_sha256: str
    registry_sha256: str
    policy_sha256: str
    diagnostic_manifest_sha256: str
    bundles: tuple[Bundle, ...]
    examples: tuple[Example, ...]
    training_eligible: bool = False
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class Decision:
    pair: PairKey
    review_status: Status
    operator_relation: Relation | None
    reviewer: str
    reviewed_at: str
    note: str
    geometry: Geometry | None = None
    visibility: Visibility | None = None

    def __post_init__(self) -> None:
        if not self.reviewer.strip() or not self.note.strip():
            raise ValueError("Enter an author and a short rationale")
        when = datetime.fromisoformat(self.reviewed_at)
        if when.utcoffset() is None:
            raise ValueError("Review time must include a time zone")
        if self.review_status == Status.ACCEPTED:
            if self.operator_relation in {None, Relation.NOT_ASSESSED}:
                raise ValueError("Accepted decision requires an explicit relation")
        elif self.operator_relation is not None:
            raise ValueError("Only accepted decisions carry an operator relation")


@dataclass(frozen=True, slots=True)
class Progress:
    batch_sha256: str
    decisions: tuple[Decision, ...]
    schema_version: int = 1

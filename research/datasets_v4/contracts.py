"""Canonical version-one contracts. Unknown facts are never synthetic IDs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
import re


def identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}", value
    ):
        raise ValueError(f"Invalid identifier: {value!r}")


def digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters")


class Role(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PROTECTED_TEST = "protected_test"
    DIAGNOSTIC = "diagnostic"
    QUARANTINE = "quarantine"


class Verification(StrEnum):
    UNKNOWN = "unknown"
    DECLARED = "declared"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"


class Method(StrEnum):
    MANIFEST = "native_manifest"
    COCO = "native_coco"
    VISUAL = "audit_visual_review"
    OPERATOR = "operator_review"
    BYTES = "exact_bytes"
    AUDIT = "audit_report"
    HEURISTIC = "visual_similarity_candidate"


class Knowledge(StrEnum):
    UNKNOWN = "unknown"
    KNOWN = "known"
    ABSENT = "confirmed_absent"


class Meaning(StrEnum):
    FOOTPRINT = "physical_slot_footprint"
    SLOT_REGION = "slot_classification_region"
    VEHICLE_MASK = "vehicle_instance_mask"
    VEHICLE_BBOX = "vehicle_bbox"
    CANOPY = "tree_canopy_mask"
    PAINT = "horizontal_marking_mask"
    PARKING_AREA = "whole_parking_area"
    OCCUPANCY = "roi_occupancy"
    VISIBILITY = "roi_visibility"
    NONPARKING = "nonparking_diagnostic_region"
    UNKNOWN = "unassessed_semantics"


class Transform(StrEnum):
    CROP = "crop"
    TILE = "spatial_tile"
    RESIZE = "resize"
    ROTATION = "rotation"
    AUGMENTATION = "augmentation"
    REENCODE = "reencode"
    COPY = "copy"
    UNSPECIFIED = "confirmed_derivation_parameters_unknown"


class RelationKind(StrEnum):
    SAME_PHOTOGRAPH = "same_photograph"
    LOCATION_DEPENDENCY = "location_dependency"
    DIFFERENT_ORIGIN = "confirmed_different_origin"


class ReviewState(StrEnum):
    PENDING = "pending"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"


@dataclass(frozen=True, slots=True)
class Asset:
    path: str
    sha256: str
    member: str | None = None

    def __post_init__(self) -> None:
        digest(self.sha256)
        if not Path(self.path).is_absolute():
            raise ValueError("Asset paths must be absolute")
        if self.member is not None and (
            not self.member
            or self.member.startswith("/")
            or ".." in Path(self.member).parts
        ):
            raise ValueError("Unsafe archive member")


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    asset: Asset
    locator: str
    method: Method
    verification: Verification

    def __post_init__(self) -> None:
        identifier(self.id)
        if not self.locator or not self.method:
            raise ValueError("Evidence needs a locator and method")


@dataclass(frozen=True, slots=True)
class Fact:
    state: Knowledge = Knowledge.UNKNOWN
    value: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state == Knowledge.KNOWN:
            if (
                not self.value
                or self.value.lower() in {"unknown", "none", "unmapped"}
                or not self.evidence
            ):
                raise ValueError("Known facts require a real value and evidence")
        elif self.value is not None:
            raise ValueError("Unknown/absent facts cannot carry a made-up value")
        if self.state == Knowledge.ABSENT and not self.evidence:
            raise ValueError("Confirmed absence requires evidence")


@dataclass(frozen=True, slots=True)
class License:
    declaration: str | None
    publisher: str | None
    verification: Verification
    local_research_authorized: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Dataset:
    id: str
    version: str | None
    source: str
    license: License
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        identifier(self.id)
        if self.version == "" or not self.source or not self.evidence:
            raise ValueError(
                "Dataset source/evidence required; unknown version is null"
            )


@dataclass(frozen=True, slots=True)
class Derivation:
    parent: str
    transform: Transform
    evidence: tuple[str, ...]
    parameters_json: str = "{}"


@dataclass(frozen=True, slots=True)
class Annotation:
    meaning: Meaning
    classes: tuple[str, ...]
    exhaustive_classes: tuple[str, ...]
    coverage: Verification
    evidence: tuple[str, ...]
    binding: Asset | None = None
    locator: str = ""
    mask_visibility: str = "unassessed"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in self.classes:
            raise ValueError("Annotation value outside its source classes")


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    dataset: str
    source_id: str
    asset: Asset | None
    provenance: Verification
    site: Fact = Fact()
    camera: Fact = Fact()
    session: Fact = Fact()
    day: Fact = Fact()
    family: Fact = Fact()
    location_group: Fact = Fact()
    parents: tuple[Derivation, ...] = ()
    annotations: tuple[Annotation, ...] = ()
    historical_role: Role = Role.QUARANTINE
    training_eligible: bool = False
    geometry_excluded: bool = False
    is_parking_slot: bool | None = None
    reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    rgb_size_sha256: str | None = None
    source_split: str | None = None

    def __post_init__(self) -> None:
        identifier(self.id)
        identifier(self.dataset)
        for fact in (
            self.site,
            self.camera,
            self.session,
            self.family,
            self.location_group,
        ):
            if fact.state == Knowledge.KNOWN:
                identifier(fact.value)
        if self.day.state == Knowledge.KNOWN:
            date.fromisoformat(self.day.value)
        if self.rgb_size_sha256 is not None:
            digest(self.rgb_size_sha256)
        if not self.source_id or not self.evidence:
            raise ValueError("Source identity and evidence required")


@dataclass(frozen=True, slots=True)
class Relation:
    id: str
    left: str
    right: str
    kind: RelationKind
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        identifier(self.id)
        if self.left == self.right or not self.evidence:
            raise ValueError("A relation needs different endpoints and evidence")


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    left: str
    right: str
    method: str
    evidence: tuple[str, ...]
    status: ReviewState = ReviewState.PENDING
    relation: str | None = None


@dataclass(frozen=True, slots=True)
class Registry:
    id: str
    datasets: tuple[Dataset, ...]
    records: tuple[Record, ...]
    evidence: tuple[Evidence, ...]
    relations: tuple[Relation, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    approvals: tuple[Approval, ...] = ()
    candidate_indices: tuple[Asset, ...] = ()
    schema_version: int = 1


class Protocol(StrEnum):
    SEMANTIC_TRANSFORM = "reviewed_semantic_transform"
    IGNORE = "reviewed_ignore_regions"
    CONDITIONAL = "reviewed_class_conditional_supervision"


@dataclass(frozen=True, slots=True)
class Approval:
    id: str
    records: tuple[str, ...]
    source: Meaning
    target: Meaning
    classes: tuple[str, ...]
    protocol: Protocol
    reviewer: str
    rationale: str
    artifact: Asset
    implementation: Asset
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    meaning: Meaning
    classes: tuple[str, ...]
    approval: str | None = None


@dataclass(frozen=True, slots=True)
class Assignment:
    record: str
    role: Role
    task: str | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    id: str
    registry_sha256: str
    policy_sha256: str
    assignments: tuple[Assignment, ...]
    tasks: tuple[Task, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class Lock:
    record: str
    role: Role
    evidence: tuple[str, ...]
    training_eligible: bool | None = None
    geometry_excluded: bool | None = None
    is_parking_slot: bool | None = None


@dataclass(frozen=True, slots=True)
class Policy:
    id: str
    locks: tuple[Lock, ...]
    protected_sites: tuple[str, ...]
    validation_only_sites: tuple[str, ...]
    protected_assets: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class Issue:
    rule: str
    records: tuple[str, ...]
    message: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    record: str
    requested_role: Role
    effective_role: Role
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    decisions: tuple[Decision, ...]
    issues: tuple[Issue, ...]
    registry_sha256: str
    policy_sha256: str

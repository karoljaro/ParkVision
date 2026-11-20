"""Geometry proposal evidence in the pixel coordinates of the reference image."""

from dataclasses import dataclass
from typing import Literal

from .line_segment import LineSegment
from .marking_pair import MarkingPair
from .segment_group import SegmentGroup
from .segment_merging import PixelPoint
from .slot_proposal import SlotProposal
from .topology_candidate import TopologyCandidates
from .topology_support import IntersectionSourceSupport


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    start: PixelPoint
    end: PixelPoint
    kind: Literal["observed", "estimated"]
    line_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    """An unapproved hypothesis. An unresolved candidate has no polygon."""

    status: Literal["observed", "estimated", "unresolved"]
    proposal: SlotProposal | None
    boundaries: tuple[BoundaryEvidence, ...]
    junction_ids: tuple[int, ...]
    reason: str
    separator_ids: tuple[int, ...] = ()
    rail_id: int | None = None
    anchor_positions: tuple[PixelPoint, ...] = ()
    geometry_kind: Literal["complete", "partial"] = "complete"
    missing_boundaries: tuple[str, ...] = ()
    row_id: int | None = None
    rejection_code: str | None = None
    rejection_line_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RowEvidence:
    """One local rail side, including rows too sparse to propose a physical bay."""

    rail_id: int | None
    side: int
    separator_ids: tuple[int, ...]
    junction_ids: tuple[int, ...]
    positions: tuple[PixelPoint, ...]
    compatible_pairs: tuple[bool, ...]
    reason: str
    geometry_kind: Literal["shared_marking", "open_ended", "partial"] = "shared_marking"
    anchor_kinds: tuple[str, ...] = ()
    anchor_line_ids: tuple[tuple[int, ...], ...] = ()
    separator_tips: tuple[PixelPoint, ...] = ()
    direction: PixelPoint | None = None
    row_id: int | None = None
    rejection_code: str | None = None


@dataclass(frozen=True, slots=True)
class GeometryInitialization:
    """A single reference frame's evidence, including hypotheses without ROI."""

    segments: tuple[LineSegment, ...]
    fitted_lines: tuple[LineSegment, ...]
    pairs: tuple[MarkingPair, ...]
    topology: TopologyCandidates
    support: tuple[IntersectionSourceSupport, ...]
    candidates: tuple[SlotCandidate, ...]
    source_groups: tuple[tuple[int, SegmentGroup], ...] = ()
    rows: tuple[RowEvidence, ...] = ()

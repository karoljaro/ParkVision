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

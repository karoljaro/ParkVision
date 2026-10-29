"""Source-fragment support measurements, separate from topology hypotheses."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SourceWindowSupport:
    """Union of raw LSD projections in a signed window relative to a hit.

    Distances are along the topology line, without lateral paint verification.
    Windows are never clipped to fitted extents. Overlaps count once; source IDs
    include only fragments with positive overlap. No gaps are filled.
    """

    window_px: tuple[float, float]
    covered_length_px: float
    source_segment_ids: tuple[int, ...]

    @property
    def coverage_ratio(self) -> float:
        return self.covered_length_px / (self.window_px[1] - self.window_px[0])


@dataclass(frozen=True, slots=True)
class EdgeSourceSupport:
    """One original edge group; both edges of a marking stay separate.

    nearest_fragment_gap_px is longitudinal distance from the exact hit to the
    nearest projected raw interval, not Euclidean distance to an edge/paint.
    Arms follow the stored topology line's start→end direction, even for context.
    """

    fitted_edge_id: int
    nearest_fragment_gap_px: float
    junction: SourceWindowSupport
    before: SourceWindowSupport
    after: SourceWindowSupport


@dataclass(frozen=True, slots=True)
class LineSourceSupport:
    line_id: int
    kind: Literal["axis", "unpaired_context"]
    edges: tuple[EdgeSourceSupport, ...]


@dataclass(frozen=True, slots=True)
class IntersectionSourceSupport:
    """Evidence for an exact pairwise hit, never for a deduplication anchor."""

    candidate_id: int
    intersection_index: int
    lines: tuple[LineSourceSupport, LineSourceSupport]

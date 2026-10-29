"""Pixel-space diagnostic models; none of these confirm painted corners."""

from dataclasses import dataclass
from typing import Literal

from .line_segment import LineSegment


@dataclass(frozen=True, slots=True)
class LineIntersection:
    """One two-line hypothesis, retained even when nearby hits share a node.

    Roles and extension lengths correspond to line_ids. A positive extension
    makes this hypothetical, even when it is smaller than the endpoint tolerance.
    Zero extension means inside fitted extents, NOT verified source coverage.
    """

    position: tuple[float, float]
    line_ids: tuple[int, int]
    topology: Literal["L", "T", "X"]
    roles: tuple[Literal["end", "interior"], Literal["end", "interior"]]
    extension_px: tuple[float, float]
    uses_unpaired_context: bool

    @property
    def hypothetical(self) -> bool:
        """Report whether either fitted line needs extension to reach this hit."""
        return any(length > 0 for length in self.extension_px)


@dataclass(frozen=True, slots=True)
class JunctionCandidate:
    """Nearby hits with an unchanged anchor and all pairwise evidence.

    Topologies belong to individual hits, not their union. Three or more lines
    can give conflicting labels; no multi-line painted junction is inferred.
    """

    candidate_id: int
    position: tuple[float, float]
    intersections: tuple[LineIntersection, ...]


@dataclass(frozen=True, slots=True)
class EndpointCandidate:
    """A fitted endpoint, including unmatched and potentially off-frame ends.

    Nearby end-role hits link to nodes without removing this endpoint or claiming
    that the physical marking ends here. Interior crossings do not consume ends.
    """

    line_id: int
    end: Literal["start", "end"]
    position: tuple[float, float]
    junction_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TopologyCandidates:
    """Keep every original axis/context object, even when it has no hits."""

    lines: tuple[LineSegment, ...]
    unpaired_line_ids: tuple[int, ...]
    junctions: tuple[JunctionCandidate, ...]
    endpoints: tuple[EndpointCandidate, ...]

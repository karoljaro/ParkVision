from dataclasses import dataclass

from .line_segment import LineSegment


@dataclass(frozen=True, slots=True)
class MarkingPair:
    """Keep both fitted edge objects and their derived, shared-extent axis.

    widths_px gives the transverse separation at the centerline's start/end.
    overlap_ratio is shared projection / longer edge projection. These are
    geometric measurements, not probabilities of paint. Source groups remain
    with the caller; each edge's ID identifies its fitted representative.
    """

    edges: tuple[LineSegment, LineSegment]
    centerline: LineSegment
    widths_px: tuple[float, float]
    overlap_ratio: float

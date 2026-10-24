from dataclasses import dataclass

from .line_segment import LineSegment


@dataclass(frozen=True, slots=True)
class SegmentGroup:
    """A group of line segments that are nearly collinear and can be merged."""
    segments: tuple[LineSegment, ...]

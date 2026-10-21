from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class LineSegment:
    """An LSD segment in pixel coordinates, with an ID local to one extraction."""

    segment_id: int
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        """Length in pixels, derived from the unrounded endpoints."""
        return hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

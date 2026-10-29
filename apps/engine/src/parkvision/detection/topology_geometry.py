"""Cached pixel geometry used internally by topology candidate detection."""

from dataclasses import dataclass

from .line_segment import LineSegment


@dataclass(frozen=True, slots=True)
class TopologyGeometry:
    """Keep a source line, unit direction, length and extended search bbox."""

    line: LineSegment
    unit: tuple[float, float]
    length: float
    bbox: tuple[float, float, float, float]

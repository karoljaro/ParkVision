from dataclasses import dataclass
from .point import Point


@dataclass(frozen=True, slots=True)
class Polygon:
    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(set(self.vertices)) < 3:
            raise ValueError(
                f"A polygon must have at least 3 vertices, got {len(self.vertices)}"
            )

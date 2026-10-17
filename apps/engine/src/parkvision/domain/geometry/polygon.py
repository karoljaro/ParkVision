from dataclasses import dataclass
from .point import Point
from .predicates import segments_intersect


@dataclass(frozen=True, slots=True)
class Polygon:
    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        unique_vertices = set(self.vertices)
        unique_vertices_count = len(unique_vertices)

        if unique_vertices_count < 3:
            raise ValueError(
                "A polygon must have at least 3 unique vertices, "
                f"got {unique_vertices_count}."
            )

        if (area := self.area) <= 1e-12:
            raise ValueError(
                "A polygon must have a non-zero area, "
                f"got {area}."
            )

        if self._has_self_intersections():
            raise ValueError("Polygon must not have self-intersections")

    @property
    def area(self) -> float:
        next_vertices = self.vertices[1:] + (self.vertices[0],)

        total = 0.0

        for p1, p2 in zip(self.vertices, next_vertices):
            total += p1.x * p2.y
            total -= p2.x * p1.y

        return abs(total) / 2.0

    def _has_self_intersections(self) -> bool:
        vertices = self.vertices
        count = len(vertices)

        edges = [
            (vertices[i], vertices[(i + 1) % count])
            for i in range(count)
        ]

        for i in range(count):
            for j in range(i + 2, count):

                if i == 0 and j == count - 1:
                    continue

                edge1 = edges[i]
                edge2 = edges[j]

                if segments_intersect(edge1[0], edge1[1], edge2[0], edge2[1]):
                    return True

        return False

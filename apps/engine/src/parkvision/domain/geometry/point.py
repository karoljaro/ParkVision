from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not (0.0 <= self.x <= 1.0):
            raise ValueError(
                f"x must be a finite number between 0 and 1, got {self.x}"
            )

        if not isfinite(self.y) or not (0.0 <= self.y <= 1.0):
            raise ValueError(
                f"y must be a finite number between 0 and 1, got {self.y}"
            )

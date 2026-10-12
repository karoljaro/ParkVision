from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Camera:
    id: str
    name: str

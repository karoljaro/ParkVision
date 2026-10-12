from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Parking:
    id: str
    name: str

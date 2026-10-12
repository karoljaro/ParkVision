from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParkingSlot:
    id: str
    label: str | None = None

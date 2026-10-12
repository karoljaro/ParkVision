from dataclasses import dataclass

from .types import ParkingId


@dataclass(frozen=True, slots=True)
class Parking:
    id: ParkingId
    name: str

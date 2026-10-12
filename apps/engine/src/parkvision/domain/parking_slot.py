from dataclasses import dataclass

from .types import ParkingSlotId, ParkingId


@dataclass(frozen=True, slots=True)
class ParkingSlot:
    id: ParkingSlotId
    parking_id: ParkingId
    label: str | None = None

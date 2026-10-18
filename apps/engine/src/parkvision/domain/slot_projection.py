from dataclasses import dataclass

from .geometry.polygon import Polygon
from .types import CameraId, ParkingSlotId


@dataclass(frozen=True, slots=True)
class SlotProjection:
    camera_id: CameraId
    parking_slot_id: ParkingSlotId
    polygon: Polygon

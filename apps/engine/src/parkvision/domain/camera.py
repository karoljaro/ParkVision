from dataclasses import dataclass

from .types import CameraId, ParkingId


@dataclass(frozen=True, slots=True)
class Camera:
    id: CameraId
    parking_id: ParkingId
    name: str

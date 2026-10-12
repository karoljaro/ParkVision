from typing import NewType

ParkingId = NewType("ParkingId", str)
CameraId = NewType("CameraId", str)
ParkingSlotId = NewType("ParkingSlotId", str)

__all__ = [
    "ParkingId",
    "CameraId",
    "ParkingSlotId",
]

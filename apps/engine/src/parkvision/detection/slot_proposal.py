from dataclasses import dataclass

from ..domain.types import CameraId
from ..domain.geometry.polygon import Polygon


@dataclass(frozen=True, slots=True)
class SlotProposal:
    camera_id: CameraId
    polygon: Polygon

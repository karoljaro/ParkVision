from typing import Protocol

from ..capture.frame import Frame
from .slot_proposal import SlotProposal


class SlotDetector(Protocol):
    def detect(self, frame: Frame) -> tuple[SlotProposal, ...]: ...

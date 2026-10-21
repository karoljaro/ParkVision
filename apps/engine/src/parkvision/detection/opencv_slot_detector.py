import cv2
import numpy as np

from ..capture.frame import Frame
from .line_segment import LineSegment
from .slot_detector import SlotDetector
from .slot_proposal import SlotProposal


class OpenCVSlotDetector(SlotDetector):
    """Coordinate the OpenCV pipeline; currently only LSD extraction is available."""

    def __init__(self) -> None:
        self._lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def extract_segments(self, frame: Frame) -> tuple[LineSegment, ...]:
        """Extract all LSD segments from a non-empty uint8 BGR frame.

        Coordinates stay in pixels and retain subpixel precision. Segment IDs
        start at zero for each call and do not identify physical markings.
        """
        image = frame.image
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Expected a non-empty image")

        # LSD performs its own smoothing; avoid an additional blur at this stage.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines, _, _, _ = self._lsd.detect(gray)
        if lines is None:
            return ()

        return tuple(
            LineSegment(
                segment_id=segment_id,
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
            )
            for segment_id, (x1, y1, x2, y2) in enumerate(lines.reshape(-1, 4))
        )

    def detect(self, frame: Frame) -> tuple[SlotProposal, ...]:
        self.extract_segments(frame)
        raise NotImplementedError(
            "Parking slot assembly is not implemented yet; "
            "use extract_segments() for LSD results"
        )

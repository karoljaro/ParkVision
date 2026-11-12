from math import hypot

import cv2
import numpy as np

from ..capture.frame import Frame
from ..domain.types import CameraId
from .initialization import GeometryInitialization
from .line_fitting import fit_group_line
from .line_segment import LineSegment
from .marking_fragments import pair_marking_fragments
from .marking_topology import find_topology_candidates
from .segment_merging import group_segments
from .slot_assembly import assemble_slots
from .slot_detector import SlotDetector
from .slot_proposal import SlotProposal
from .topology_support_measurement import measure_topology_support


class OpenCVSlotDetector(SlotDetector):
    """Generate unapproved whole-slot proposals from a reference frame."""

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
        return tuple(
            candidate.proposal
            for candidate in self.analyze(frame).candidates
            if candidate.proposal is not None
        )

    def analyze(self, frame: Frame) -> GeometryInitialization:
        """Retain the evidence needed to review detected and unresolved slots."""
        segments = self.extract_segments(frame)
        height, width = frame.image.shape[:2]
        scale = max(0.5, hypot(width, height) / 1440.0)
        # Fitted extents can bridge a short gap; observations use source fragments.
        groups = group_segments(
            tuple(segment for segment in segments if segment.length >= 7.5 * scale),
            angle_tolerance=3.0,
            distance_tolerance=max(1.25, min(4.0, 2.9 * scale)),
            gap_tolerance=max(20.0, min(40.0, 33.3 * scale)),
        )
        first_id = max((segment.segment_id for segment in segments), default=-1) + 1
        fitted = tuple(
            fit_group_line(group, segment_id=first_id + offset)
            for offset, group in enumerate(groups)
        )
        sources = {line.segment_id: group for line, group in zip(fitted, groups)}
        fitted, sources, pairs = pair_marking_fragments(fitted, sources, scale)
        paired_ids = {edge.segment_id for pair in pairs for edge in pair.edges}
        topology = find_topology_candidates(
            pairs,
            tuple(line for line in fitted if line.segment_id not in paired_ids),
            endpoint_tolerance_px=5 * scale,
            max_extension_px=14 * scale,
            dedup_radius_px=5 * scale,
            min_context_length_px=25 * scale,
        )
        support = measure_topology_support(
            topology,
            pairs,
            sources,
            junction_radius_px=5 * scale,
            arm_length_px=18 * scale,
        )
        candidates = assemble_slots(
            CameraId(frame.camera_id), width, height, pairs, topology, support, sources
        )
        return GeometryInitialization(
            segments,
            fitted,
            pairs,
            topology,
            support,
            candidates,
            tuple(sources.items()),
        )

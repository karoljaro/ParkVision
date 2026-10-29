import datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from parkvision.capture.frame import Frame
from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector
from parkvision.detection.segment_merging import group_segments
from parkvision.detection.slot_detector import SlotDetector


def make_frame(image: NDArray[np.uint8]) -> Frame:
    return Frame(
        camera_id="test-camera",
        sequence=0,
        captured_at=datetime.datetime.now(datetime.timezone.utc),
        image=image,
    )


@pytest.fixture
def marked_frame() -> Frame:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(image, (40, 60), (260, 60), (255, 255, 255), 4)
    cv2.line(image, (100, 150), (118, 150), (255, 255, 255), 3)
    return make_frame(image)


def test_blank_frame_has_no_segments() -> None:
    frame = make_frame(np.zeros((120, 160, 3), dtype=np.uint8))

    assert OpenCVSlotDetector().extract_segments(frame) == ()


def test_extracts_long_and_short_markings_in_pixel_coordinates(
    marked_frame: Frame,
) -> None:
    segments = OpenCVSlotDetector().extract_segments(marked_frame)

    assert isinstance(segments, tuple)
    assert any(
        segment.length > 180
        and abs(segment.start[1] - 60) < 5
        and abs(segment.end[1] - 60) < 5
        for segment in segments
    )
    assert any(
        segment.length < 30
        and abs(segment.start[1] - 150) < 5
        and abs(segment.end[1] - 150) < 5
        for segment in segments
    )
    assert all(
        type(coordinate) is float
        for segment in segments
        for coordinate in (*segment.start, *segment.end)
    )
    assert any(
        not coordinate.is_integer()
        for segment in segments
        for coordinate in (*segment.start, *segment.end)
    )


def test_segment_ids_restart_for_each_extraction(marked_frame: Frame) -> None:
    detector = OpenCVSlotDetector()

    first = detector.extract_segments(marked_frame)
    second = detector.extract_segments(marked_frame)

    assert first
    assert second
    assert [segment.segment_id for segment in first] == list(range(len(first)))
    assert [segment.segment_id for segment in second] == list(range(len(second)))


def test_extracted_segments_can_be_grouped_and_fitted(marked_frame: Frame) -> None:
    sources = OpenCVSlotDetector().extract_segments(marked_frame)
    groups = group_segments(sources, 3.0, 2.0, 12.0)
    first_id = max(s.segment_id for s in sources) + 1

    fitted = tuple(
        fit_group_line(group, segment_id=first_id + i) for i, group in enumerate(groups)
    )

    # Check useful geometry, without depending on exact LSD fragment counts/values.
    assert any(
        line.length > 180 and abs(line.start[1] - 60) < 5 and abs(line.end[1] - 60) < 5
        for line in fitted
    )
    assert len(fitted) == len(groups)
    assert all(line.length > 0 for line in fitted)
    assert {line.segment_id for line in fitted}.isdisjoint(
        s.segment_id for s in sources
    )
    assert all(
        any(s is member for group in groups for member in group.segments)
        for s in sources
    )


def test_extraction_preserves_input_and_writes_no_files(
    marked_frame: Frame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = marked_frame.image.copy()
    monkeypatch.chdir(tmp_path)

    OpenCVSlotDetector().extract_segments(marked_frame)

    np.testing.assert_array_equal(marked_frame.image, original)
    assert list(tmp_path.iterdir()) == []


def test_length_uses_unrounded_endpoints() -> None:
    segment = LineSegment(segment_id=0, start=(12.1, 18.1), end=(12.4, 18.5))

    assert segment.length == pytest.approx(0.5)


def test_rejects_non_uint8_images() -> None:
    image = cast(NDArray[np.uint8], np.zeros((120, 160, 3), dtype=np.float32))

    with pytest.raises(TypeError, match="Expected uint8"):
        OpenCVSlotDetector().extract_segments(make_frame(image))


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(np.zeros((0, 160, 3), dtype=np.uint8), id="empty-height"),
        pytest.param(np.zeros((120, 0, 3), dtype=np.uint8), id="empty-width"),
        pytest.param(np.zeros((120, 160), dtype=np.uint8), id="grayscale"),
        pytest.param(np.zeros((120, 160, 4), dtype=np.uint8), id="four-channels"),
    ],
)
def test_rejects_unsupported_image_shapes(image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError):
        OpenCVSlotDetector().extract_segments(make_frame(image))


@pytest.mark.parametrize("marked", [False, True], ids=["blank", "marked"])
def test_slot_detection_explicitly_remains_unimplemented(
    marked: bool,
    marked_frame: Frame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = (
        marked_frame if marked else make_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    )
    original = frame.image.copy()
    monkeypatch.chdir(tmp_path)
    detector: SlotDetector = OpenCVSlotDetector()

    with pytest.raises(NotImplementedError, match="Parking slot assembly"):
        detector.detect(frame)

    np.testing.assert_array_equal(frame.image, original)
    assert list(tmp_path.iterdir()) == []

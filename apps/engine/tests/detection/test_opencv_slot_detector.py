import datetime
from math import dist
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
def test_insufficient_evidence_returns_no_polygon(
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

    assert detector.detect(frame) == ()

    np.testing.assert_array_equal(frame.image, original)
    assert list(tmp_path.iterdir()) == []


def test_actual_lsd_recovers_adjacent_perspective_slots(
    geometry_row_image: NDArray[np.uint8],
) -> None:
    frame = make_frame(geometry_row_image)
    detector = OpenCVSlotDetector()
    result = detector.analyze(frame)
    proposals = detector.detect(frame)

    assert len(proposals) == 3
    assert len(result.pairs) >= 6
    assert len(result.topology.junctions) >= 6
    assert result.support
    assert sum(candidate.status == "observed" for candidate in result.candidates) >= 2
    assert all(
        len(candidate.junction_ids) == 4
        for candidate in result.candidates
        if candidate.proposal is not None
    )
    assert all(proposal.camera_id == frame.camera_id for proposal in proposals)
    assert all(len(proposal.polygon.vertices) == 4 for proposal in proposals)
    assert all(
        0 <= coordinate <= 1
        for proposal in proposals
        for point in proposal.polygon.vertices
        for coordinate in (point.x, point.y)
    )
    # The left-hand slot is visibly wider at the near rail.
    left = min(proposals, key=lambda p: min(v.x for v in p.polygon.vertices))
    vertices = left.polygon.vertices
    horizontal_widths = sorted(
        abs(start.x - end.x) * (geometry_row_image.shape[1] - 1)
        for start, end in zip(vertices, vertices[1:] + vertices[:1])
        if abs(start.y - end.y) < 0.02
    )
    assert horizontal_widths == pytest.approx([75, 91], abs=2)
    assert detector.detect(frame) == proposals


def test_isolated_vehicle_like_outline_is_insufficient() -> None:
    image = np.full((260, 440, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (110, 65), (275, 210), (230, 230, 230), thickness=5)
    result = OpenCVSlotDetector().analyze(make_frame(image))
    assert result.segments
    assert all(
        item.status == "unresolved" and item.proposal is None
        for item in result.candidates
    )


def test_dark_strip_edges_do_not_establish_painted_separators(
    geometry_row_image: NDArray[np.uint8],
) -> None:
    detector = OpenCVSlotDetector()
    result = detector.analyze(make_frame(cv2.bitwise_not(geometry_row_image)))
    assert result.segments
    assert all(candidate.proposal is None for candidate in result.candidates)


def test_cctv_pilot_recovers_three_individual_left_bays() -> None:
    image = cv2.imread(str(Path(__file__).resolve().parents[2] / "samples/parking.jpg"))
    height, width = image.shape[:2]
    result = OpenCVSlotDetector().analyze(make_frame(image))
    polygons = [
        np.array(
            [
                (point.x * (width - 1), point.y * (height - 1))
                for point in candidate.proposal.polygon.vertices
            ],
            dtype=np.float32,
        )
        for candidate in result.candidates
        if candidate.proposal is not None
    ]
    # Cropped to the three operator-reviewed bays; tolerances are for this
    # development image and allow variation in subpixel LSD fits.
    separators = (
        ((32.5, 408), (480.5, 280)),
        ((66.5, 461), (539.5, 312.25)),
        ((109, 525.5), (609.5, 350.5)),
        ((161.5, 605.5), (690, 394.25)),
    )
    matches = []
    for (left_a, right_a), (left_b, right_b) in zip(separators, separators[1:]):
        expected = np.array((left_a, right_a, right_b, left_b), dtype=np.float32)
        ious = []
        for polygon in polygons:
            overlap, _ = cv2.intersectConvexConvex(polygon, expected)
            union = cv2.contourArea(polygon) + cv2.contourArea(expected) - overlap
            ious.append(overlap / union)
        best = max(range(len(ious)), key=ious.__getitem__)
        assert ious[best] >= 0.86
        matches.append(best)
    assert len(set(matches)) == 3
    # The ladder marking at the lower right is not a row of parking bays.
    assert all(
        cv2.pointPolygonTest(polygon, (1343, 793), False) < 0 for polygon in polygons
    )
    for index, polygon in enumerate(polygons):
        for other in polygons[index + 1 :]:
            overlap, _ = cv2.intersectConvexConvex(polygon, other)
            assert overlap <= 0.10 * min(
                cv2.contourArea(polygon), cv2.contourArea(other)
            )


def test_actual_lsd_marks_a_short_occluded_separator_span() -> None:
    image = np.full((260, 440, 3), 35, dtype=np.uint8)
    paint = (230, 230, 230)
    for y in (44, 203):
        cv2.line(image, (35, y), (390, y), paint, 3)
    for x in (72, 154, 242, 327):
        cv2.line(image, (x, 44), (x, 203), paint, 3)
    cv2.rectangle(image, (140, 111), (170, 120), (35, 35, 35), -1)

    result = OpenCVSlotDetector().analyze(make_frame(image))
    assert sum(item.proposal is not None for item in result.candidates) == 3
    gaps = [
        span
        for item in result.candidates
        if item.proposal is not None
        for span in item.boundaries
        if span.kind == "estimated"
        and len(span.line_ids) == 1
        and 100
        < min(span.start[1], span.end[1])
        < max(span.start[1], span.end[1])
        < 130
    ]
    assert len(gaps) == 2
    assert gaps[0] == gaps[1]
    assert 9 < dist(gaps[0].start, gaps[0].end) < 18

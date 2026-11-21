from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import cv2
import numpy as np
import pytest

from parkvision.capture.occupancy_source import (
    RTSP_CLOCK,
    VIDEO_CLOCK,
    RTSPSource,
    VideoFileSource,
)

ANCHOR = datetime.fromisoformat("2026-11-20T12:00:00+01:00")
IMAGE = np.zeros((48, 64, 3), dtype=np.uint8)


def test_video_file_sampling_uses_decoded_pts_without_a_system_time(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full_like(IMAGE, i * 30))
    writer.release()

    source = VideoFileSource(path, "cam-1", ANCHOR, sample_hz=5)
    events = list(source)
    frames = events[:-1]
    assert [event.kind for event in frames] == ["frame", "frame", "frame"]
    assert [event.frame.sequence for event in frames] == [0, 2, 4]
    assert [event.pts_seconds for event in frames] == pytest.approx([0, 0.2, 0.4])
    assert [event.skipped_frames for event in frames] == [0, 1, 1]
    assert [event.frame.captured_at for event in frames] == [
        ANCHOR + timedelta(seconds=offset) for offset in (0, 0.2, 0.4)
    ]
    assert {event.time_source for event in events} == {VIDEO_CLOCK}
    assert {event.frame.camera_id for event in frames} == {"cam-1"}
    assert all(event.decode_seconds >= 0 for event in events)
    assert events[-1].kind == "eof"
    assert events[-1].skipped_frames == 1


class VideoCaptureFixture:
    """Simulate only the VideoCapture methods used by the source."""

    def __init__(self, pts_ms, *, expected=None, image=IMAGE):
        self._pts = iter(pts_ms)
        self._current_pts = None
        self._expected = len(pts_ms) if expected is None else expected
        self.image = image
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._expected
        assert prop == cv2.CAP_PROP_POS_MSEC
        return self._current_pts

    def read(self):
        self._current_pts = next(self._pts, None)
        if self._current_pts is None:
            return False, None
        return True, self.image

    def release(self):
        self.released = True


@pytest.mark.parametrize("nonmonotonic_or_invalid", [200.0, 199.0, float("nan")])
def test_invalid_pts_stops_video_instead_of_guessing_a_timestamp(
    tmp_path, monkeypatch, nonmonotonic_or_invalid
):
    path = tmp_path / "fixture.avi"
    path.touch()
    capture = VideoCaptureFixture([10.0, 90.0, 200.0, nonmonotonic_or_invalid])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    events = list(VideoFileSource(path, "cam-1", ANCHOR))
    assert [event.pts_seconds for event in events[:-1]] == [0.01, 0.09, 0.2]
    assert events[-1].kind == "disconnected"
    assert events[-1].reason == "invalid_video_pts"
    assert events[-1].skipped_frames == 1
    assert capture.released


def test_video_corruption_before_reported_eof_is_not_a_clean_eof(tmp_path, monkeypatch):
    path = tmp_path / "fixture.avi"
    path.touch()
    capture = VideoCaptureFixture([0.0, 100.0], expected=3)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    events = list(VideoFileSource(path, "cam-1", ANCHOR))
    assert events[-1].kind == "disconnected"
    assert events[-1].reason == "video_decode_failed"
    assert capture.released


def test_video_missing_input_and_naive_absolute_anchor_are_rejected(tmp_path):
    path = tmp_path / "missing.avi"
    with pytest.raises(ValueError, match="UTC offset"):
        VideoFileSource(path, "cam-1", ANCHOR.replace(tzinfo=None))
    (event,) = list(VideoFileSource(path, "cam-1", ANCHOR))
    assert event.kind == "disconnected"
    assert event.reason == "video_open_failed"


def test_video_rejects_decoding_without_bgr_pixels(tmp_path, monkeypatch):
    path = tmp_path / "fixture.avi"
    path.touch()
    capture = VideoCaptureFixture([0], image=np.zeros((48, 64), dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    (event,) = list(VideoFileSource(path, "cam-1", ANCHOR))
    assert event.kind == "disconnected" and event.reason == "invalid_image"
    assert capture.released


class BlockingCapture:
    def __init__(self, frames, ready_to_fail):
        self.remaining = frames
        self.ready_to_fail = ready_to_fail
        self.drained = Event()
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.remaining:
            self.remaining -= 1
            return True, IMAGE
        self.drained.set()
        self.ready_to_fail.wait(2)
        return False, None

    def release(self):
        self.released = True


def _next_kind(iterator, kind):
    for _ in range(50):
        event = next(iterator)
        if event.kind == kind:
            return event
        assert event.kind == "heartbeat"
    raise AssertionError(f"Expected {kind}")


def test_rtsp_reports_missing_signal_and_recovers_with_increasing_sequence(monkeypatch):
    loss = Event()
    stop = Event()
    captures = [BlockingCapture(1, loss), BlockingCapture(1, stop)]
    expected = iter(captures)

    def factory(url, backend, params):
        assert url == "rtsp://camera.example/stream"
        assert backend == cv2.CAP_FFMPEG
        assert params == [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            500,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            500,
        ]
        return next(expected)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    source = RTSPSource(
        "rtsp://camera.example/stream",
        "cam-1",
        timeout_seconds=0.5,
        reconnect_seconds=0.01,
        heartbeat_seconds=0.02,
    )
    iterator = iter(source)
    try:
        first = _next_kind(iterator, "frame")
        assert source.connected and source.is_current(first)
        assert next(iterator).kind == "heartbeat"
        loss.set()
        disconnected = _next_kind(iterator, "disconnected")
        second = _next_kind(iterator, "frame")
        assert disconnected.reason == "rtsp_signal_lost"
        assert first.frame.sequence == 0 and second.frame.sequence == 1
        assert first.frame.captured_at < second.frame.captured_at
        assert second.frame.captured_at.tzinfo == timezone.utc
        assert second.time_source == RTSP_CLOCK and second.pts_seconds is None
        assert source.connected and source.is_current(second)
        assert second.generation != first.generation and not source.is_current(first)
    finally:
        source._stopped.set()
        loss.set()
        stop.set()
        iterator.close()
    assert all(capture.released for capture in captures)


def test_rtsp_only_retains_the_latest_frame_and_reports_dropped_frames(monkeypatch):
    stop = Event()
    capture = BlockingCapture(10, stop)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    source = RTSPSource(
        "rtsp://camera.example/stream",
        "cam-1",
        timeout_seconds=0.5,
        heartbeat_seconds=0.02,
    )
    iterator = iter(source)
    try:
        first = _next_kind(iterator, "frame")
        assert capture.drained.wait(1)
        if first.frame.sequence == 9:
            last = first
            assert first.skipped_frames == 9
        else:
            last = _next_kind(iterator, "frame")
            assert first.skipped_frames + last.skipped_frames == 8
        assert last.frame.sequence == 9
    finally:
        # Release the test backend while close is waiting on the capture thread.
        releaser = Thread(target=lambda: (source._stopped.wait(1), stop.set()))
        releaser.start()
        iterator.close()
        releaser.join()
    assert capture.released

"""OpenCV sources for occupancy, with distinct clocks for files and RTSP."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from threading import Condition, Event, Thread
from time import monotonic, perf_counter
from typing import Literal
from urllib.parse import urlsplit

import cv2
import numpy as np

from .frame import Frame

VIDEO_CLOCK = "explicit_anchor_plus_video_pts"
RTSP_CLOCK = "host_utc_receipt"


@dataclass(frozen=True, slots=True)
class FrameEvent:
    kind: Literal["frame", "heartbeat", "disconnected", "eof"]
    frame: Frame | None = None
    pts_seconds: float | None = None
    skipped_frames: int = 0
    decode_seconds: float = 0.0
    reason: str | None = None
    time_source: str | None = None
    generation: int | None = None


def _validate_rate(sample_hz: float | None) -> None:
    if sample_hz is not None and (not isfinite(sample_hz) or sample_hz <= 0):
        raise ValueError("sample_hz must be positive and finite")


def _valid_image(image: object) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and image.size > 0
    )


class VideoFileSource:
    """Read presentation timestamps in decode order; no FPS-based time fallback.

    capture_start denotes PTS zero. Processing is synchronous, so a slow
    consumer leaves unread frames in the file rather than in a memory queue.
    """

    time_source = VIDEO_CLOCK

    def __init__(
        self,
        path: Path,
        camera_id: str,
        capture_start: datetime,
        *,
        sample_hz: float | None = None,
    ) -> None:
        if capture_start.utcoffset() is None:
            raise ValueError("Video capture_start must include a UTC offset")
        _validate_rate(sample_hz)
        self.path = Path(path)
        self.camera_id = camera_id
        self.capture_start = capture_start.astimezone(timezone.utc)
        self.sample_hz = sample_hz
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def __iter__(self) -> Iterator[FrameEvent]:
        if self._closed:
            return
        if not self.path.is_file():
            yield FrameEvent(
                "disconnected", reason="video_open_failed", time_source=VIDEO_CLOCK
            )
            return
        capture = None
        try:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                yield FrameEvent(
                    "disconnected", reason="video_open_failed", time_source=VIDEO_CLOCK
                )
                return
            expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            interval = 1.0 / self.sample_hz if self.sample_hz is not None else 0.0
            previous_pts = -1.0
            previous_capture: datetime | None = None
            next_sample = 0.0
            sequence = 0
            skipped = 0
            decode_seconds = 0.0
            while not self._closed:
                started = perf_counter()
                ok, image = capture.read()
                decode_seconds += perf_counter() - started
                if not ok:
                    interrupted = sequence == 0 or (
                        isfinite(expected) and expected > sequence
                    )
                    yield FrameEvent(
                        "disconnected" if interrupted else "eof",
                        skipped_frames=skipped,
                        decode_seconds=decode_seconds,
                        reason="video_decode_failed" if interrupted else None,
                        time_source=VIDEO_CLOCK,
                    )
                    return
                pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                reason = None
                if not _valid_image(image):
                    reason = "invalid_image"
                elif not isfinite(pts) or pts < 0 or pts <= previous_pts:
                    reason = "invalid_video_pts"
                else:
                    try:
                        captured_at = self.capture_start + timedelta(seconds=pts)
                    except OverflowError:
                        reason = "invalid_video_pts"
                    else:
                        if (
                            previous_capture is not None
                            and captured_at <= previous_capture
                        ):
                            reason = "invalid_video_pts"
                if reason:
                    yield FrameEvent(
                        "disconnected",
                        skipped_frames=skipped + 1,
                        decode_seconds=decode_seconds,
                        reason=reason,
                        time_source=VIDEO_CLOCK,
                    )
                    return
                previous_pts = pts
                previous_capture = captured_at
                if pts + 1e-9 >= next_sample:
                    yield FrameEvent(
                        "frame",
                        Frame(self.camera_id, sequence, captured_at, image),
                        pts_seconds=pts,
                        skipped_frames=skipped,
                        decode_seconds=decode_seconds,
                        time_source=VIDEO_CLOCK,
                    )
                    next_sample = pts + interval
                    skipped = 0
                    decode_seconds = 0.0
                else:
                    skipped += 1
                sequence += 1
        except cv2.error:
            yield FrameEvent(
                "disconnected", reason="video_decode_failed", time_source=VIDEO_CLOCK
            )
        finally:
            if capture is not None:
                capture.release()
            self._closed = True


class RTSPSource:
    """Use FFmpeg deadlines and retain at most one image between consumer events.

    This is the host's UTC receive clock, not the camera's exposure clock.
    Backend/server buffering before read() returns is outside this queue.
    """

    time_source = RTSP_CLOCK

    def __init__(
        self,
        url: str,
        camera_id: str,
        *,
        sample_hz: float | None = None,
        timeout_seconds: float = 3.0,
        reconnect_seconds: float = 1.0,
        heartbeat_seconds: float = 0.25,
    ) -> None:
        try:
            parsed = urlsplit(url)
            valid_url = parsed.scheme == "rtsp" and bool(parsed.hostname)
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("Expected an rtsp:// URL with a host")
        _validate_rate(sample_hz)
        for seconds in (timeout_seconds, reconnect_seconds, heartbeat_seconds):
            if not isfinite(seconds) or seconds <= 0:
                raise ValueError(
                    "RTSP timeout, reconnect and heartbeat must be positive"
                )
        if timeout_seconds > 30:
            raise ValueError("RTSP open and read timeouts must not exceed 30 seconds")
        self._url = url
        self.camera_id = camera_id
        self._timeout_ms = max(1, round(timeout_seconds * 1000))
        self._reconnect_seconds = reconnect_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._interval = 1.0 / sample_hz if sample_hz is not None else 0.0
        self._stopped = Event()
        self._changed = Condition()
        self._thread: Thread | None = None
        self._latest: FrameEvent | None = None
        self._loss: tuple[str, int] | None = None
        self._generation = 0
        self._last_receive: float | None = None

    def _connection_healthy(self) -> bool:
        return (
            not self._stopped.is_set()
            and self._last_receive is not None
            and monotonic() - self._last_receive < self._timeout_ms / 1000
        )

    @property
    def connected(self) -> bool:
        with self._changed:
            return self._connection_healthy()

    def is_current(self, event: FrameEvent) -> bool:
        """Check that an in-flight result still belongs to a healthy connection."""
        with self._changed:
            return (
                event.frame is not None
                and event.generation == self._generation
                and self._connection_healthy()
            )

    def _produce(self) -> None:
        sequence = 0
        previous_receipt: datetime | None = None
        while not self._stopped.is_set():
            failure = "rtsp_open_failed"
            capture = None
            try:
                capture = cv2.VideoCapture(
                    self._url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        self._timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        self._timeout_ms,
                    ],
                )
                if capture.isOpened():
                    failure = "rtsp_signal_lost"
                    next_sample = 0.0
                    while not self._stopped.is_set():
                        started = perf_counter()
                        ok, image = capture.read()
                        decode_seconds = perf_counter() - started
                        if not ok:
                            break
                        sequence += 1
                        if not _valid_image(image):
                            failure = "invalid_image"
                            break
                        receipt = datetime.now(timezone.utc)
                        if previous_receipt is not None and receipt <= previous_receipt:
                            failure = "nonmonotonic_receive_time"
                            break
                        previous_receipt = receipt
                        now = monotonic()
                        with self._changed:
                            self._last_receive = now
                            if now < next_sample:
                                continue
                            next_sample = now + self._interval
                            self._latest = FrameEvent(
                                "frame",
                                Frame(self.camera_id, sequence - 1, receipt, image),
                                decode_seconds=decode_seconds,
                                time_source=RTSP_CLOCK,
                                generation=self._generation,
                            )
                            self._changed.notify()
            except cv2.error:
                # OpenCV errors may contain the source URL, including credentials.
                pass
            finally:
                if not self._stopped.is_set():
                    with self._changed:
                        self._latest = None
                        self._loss = (failure, sequence)
                        self._generation += 1
                        self._last_receive = None
                        self._changed.notify()
                if capture is not None:
                    capture.release()
            if self._stopped.wait(self._reconnect_seconds):
                break

    def close(self) -> None:
        self._stopped.set()
        with self._changed:
            self._changed.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_ms / 1000 + 0.25)

    def __iter__(self) -> Iterator[FrameEvent]:
        if self._stopped.is_set():
            return
        if self._thread is not None:
            raise RuntimeError("RTSPSource has one consumer")
        self._thread = Thread(target=self._produce, daemon=True)
        self._thread.start()
        accounted = 0
        try:
            while not self._stopped.is_set():
                with self._changed:
                    self._changed.wait_for(
                        lambda: self._stopped.is_set()
                        or self._loss is not None
                        or self._latest is not None,
                        timeout=self._heartbeat_seconds,
                    )
                    if self._stopped.is_set():
                        break
                    if self._loss is not None:
                        reason, count = self._loss
                        self._loss = None
                        event = FrameEvent(
                            "disconnected",
                            skipped_frames=count - accounted,
                            reason=reason,
                            time_source=RTSP_CLOCK,
                        )
                        accounted = count
                    elif self._latest is not None:
                        event = self._latest
                        self._latest = None
                        assert event.frame is not None
                        last = event.frame.sequence
                        event = replace(event, skipped_frames=last - accounted)
                        accounted = last + 1
                    else:
                        event = FrameEvent("heartbeat", time_source=RTSP_CLOCK)
                yield event
        finally:
            self.close()

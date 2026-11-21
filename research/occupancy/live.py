"""Run one authorized video or RTSP source against a frozen parking configuration."""

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import signal
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np

from .live_output import LiveOutput
from .models import CONVNEXT_SHA256, MODEL_CHOICES, SLOT_SHA256, load_models
from .observe import (
    ObservationRequest,
    ObservationSetup,
    observe_pixels,
    positive_seconds,
    timestamp,
)
from .replay import observation_from_document, observation_model_identity

from parkvision.domain.temporal_occupancy import (
    StreamKey,
    TemporalPolicy,
    disconnect,
    initial_state,
    observe_slot,
    reset_stream,
    tick,
)

CHECKPOINTS = {"acpds-rcnn": SLOT_SHA256, "convnext-roi4": CONVNEXT_SHA256}


def utc_now():
    return datetime.now(timezone.utc)


class LiveSession:
    """Keep one backend and one approved view pinned until the process restarts."""

    def __init__(self, setup, parking_id, session_id, backend, diagnostic_slot_ids=()):
        configuration = setup.configuration
        if not configuration.is_approved or configuration.parking.id != parking_id:
            raise ValueError("Approved configuration does not match the parking ID")
        if not session_id or session_id != session_id.strip():
            raise ValueError("Session ID must be nonempty and have no edge whitespace")
        if backend not in CHECKPOINTS:
            raise ValueError(f"Unknown occupancy model: {backend}")
        self.setup = setup
        self.requested_session_id = session_id
        self.run_id = uuid4().hex
        self.generation = 0
        self.backend = backend
        self.checkpoint = CHECKPOINTS[backend]
        self.policy = TemporalPolicy()
        projected = {p.parking_slot_id for p in setup.projections}
        diagnostic = set(diagnostic_slot_ids)
        if not diagnostic <= projected:
            raise ValueError("Diagnostic slot IDs must have projections in this camera")
        if projected != {slot.id for slot in configuration.slots}:
            raise ValueError("Every slot must have a projection in the selected camera")
        if not projected - diagnostic:
            raise ValueError("At least one physical parking slot is required")
        self.diagnostic = diagnostic
        self.states = {}
        self.raw = {}
        self.model_provenance = None
        self.last_frame_at = None
        self.last_sequence = None
        self.latest_pts = None

    @property
    def session_id(self):
        return f"{self.requested_session_id}/{self.run_id}/{self.generation}"

    def key(self, slot_id):
        return StreamKey(
            self.setup.configuration.parking.id,
            self.setup.camera_id,
            slot_id,
            self.setup.configuration.revision,
            self.setup.configuration_sha256,
            self.session_id,
        )

    def start(self, at):
        self.states = {
            p.parking_slot_id: initial_state(self.key(p.parking_slot_id), at)
            for p in self.setup.projections
        }
        self.updated_at = at

    def reset(self, at):
        self.generation += 1
        self.states = {
            slot: reset_stream(state, self.key(slot), at)
            for slot, state in self.states.items()
        }
        self.raw = {}
        self.last_frame_at = self.last_sequence = self.latest_pts = None
        self.updated_at = at

    def check_model(self, provenance):
        identity = observation_model_identity(
            {"provenance": {"selected_model": self.backend, "models": provenance}}
        )
        if identity.get("per_slot_checkpoint") != self.checkpoint:
            raise ValueError(
                "Loaded model does not match the pinned occupancy checkpoint"
            )

    def advance(self, at, *, disconnected=False):
        self.states = {
            slot: (
                disconnect(state, at) if disconnected else tick(state, at, self.policy)
            )
            for slot, state in self.states.items()
        }
        if disconnected:
            self.raw = {}
            self.last_frame_at = self.latest_pts = None
        self.updated_at = at

    def apply(self, document, at, pts_seconds):
        provenance = document["provenance"]
        if provenance.get("selected_model") != self.backend:
            raise ValueError("Mixed occupancy backend")
        if provenance["models"] is not None:
            self.check_model(provenance["models"])
        observations = {
            slot: observation_from_document(
                document, slot, f"{self.session_id}/{document['sequence']}"
            )
            for slot in self.states
        }
        changes = {
            slot: observe_slot(self.states[slot], obs, at, self.policy)
            for slot, obs in observations.items()
        }
        if any(change.disposition != "accepted" for change in changes.values()):
            raise ValueError("Nonmonotonic frame in the active session")
        self.states = {slot: change.state for slot, change in changes.items()}
        self.raw = {row["slot_id"]: row for row in document["observations"]}
        self.last_frame_at = next(iter(observations.values())).captured_at
        self.last_sequence = document["sequence"]
        self.latest_pts = pts_seconds
        self.updated_at = at

    def snapshot(self, *, source, published_at, skipped_frames):
        counts = {
            "physical_total": 0,
            "confirmed_free": 0,
            "confirmed_occupied": 0,
            "held": 0,
            "unknown": 0,
            "stale": 0,
        }
        slots = []
        for slot, state in self.states.items():
            diagnostic = slot in self.diagnostic
            available = state.status == "confirmed" and state.estimated_state == "free"
            if not diagnostic:
                counts["physical_total"] += 1
                category = (
                    f"confirmed_{state.estimated_state}"
                    if state.status == "confirmed"
                    else state.status
                )
                counts[category] += 1
            expiry = None
            if state.status == "confirmed":
                expiry = min(
                    state.last_capture_at
                    + timedelta(seconds=self.policy.max_gap_seconds),
                    state.supporting_evidence.captured_at
                    + timedelta(seconds=self.policy.max_observation_age_seconds),
                )
            evidence = state.last_evidence
            raw = self.raw.get(slot)
            slots.append(
                {
                    "slot_id": slot,
                    "diagnostic_only": diagnostic,
                    "raw_state": raw["state"] if raw else None,
                    "unknown_reason": raw["unknown_reason"] if raw else None,
                    "estimated_state": state.estimated_state,
                    "temporal_status": state.status,
                    "temporal_reason": state.reason,
                    "last_evidence_age_seconds": (
                        (self.updated_at - evidence.captured_at).total_seconds()
                        if evidence
                        else None
                    ),
                    "confirmed_evidence_expires_at": iso(expiry),
                    "confirmed_available": available and not diagnostic,
                }
            )
        return {
            "schema_version": 1,
            "kind": "parkvision_live_occupancy",
            "parking_id": self.setup.configuration.parking.id,
            "camera_id": self.setup.camera_id,
            "session_id": self.session_id,
            "requested_session_id": self.requested_session_id,
            "run_id": self.run_id,
            "configuration": {
                "revision": self.setup.configuration.revision,
                "sha256": self.setup.configuration_sha256,
            },
            "model": {
                "backend": self.backend,
                "checkpoint_sha256": self.checkpoint,
                "provenance": self.model_provenance,
            },
            "source": source,
            "captured_at": iso(self.last_frame_at),
            "source_pts_seconds": self.latest_pts,
            "sequence": self.last_sequence,
            "updated_at": iso(self.updated_at),
            "published_at": iso(published_at),
            "skipped_frames": skipped_frames,
            "availability_counts": counts,
            "slots": slots,
        }


def iso(value):
    return value.isoformat() if value is not None else None


def run_live(
    session,
    source,
    output,
    model_loader,
    *,
    source_id,
    video_anchor=None,
    view_confirmed,
    view_check=None,
    sample_hz=1.0,
    clock=utc_now,
):
    """Apply the temporal domain to decoded frames, one selected frame at a time."""
    live = video_anchor is None
    if not live and video_anchor.utcoffset() is None:
        raise ValueError("Video anchor must include a UTC offset")
    if not view_confirmed:
        raise ValueError("The source must be explicitly matched to the approved view")
    if not isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    session.start(clock() if live else video_anchor)
    skipped = processed = event_index = 0
    discarded = 0
    health = "starting"
    geometry_valid = True
    needs_reset = False
    last_heartbeat = None
    heartbeat_after = min(1.5 / sample_hz, session.policy.max_gap_seconds)
    source_is_current = getattr(source, "is_current", None)
    model_load_seconds = None
    started = perf_counter()
    timing_totals = dict(decode=0.0, inference_pipeline=0.0, temporal=0.0, publish=0.0)

    def at_now():
        return clock() if live else session.updated_at

    def publish(kind, reason=None, **fields):
        nonlocal event_index
        snapshot = session.snapshot(
            source={
                "id": source_id,
                "kind": "rtsp" if live else "video",
                "time_source": source.time_source,
                "health": health,
                "reason": reason,
            },
            published_at=clock(),
            skipped_frames=skipped,
        )
        event = {
            "schema_version": 1,
            "event": kind,
            "event_index": event_index,
            **fields,
            "snapshot": snapshot,
        }
        before = perf_counter()
        output.publish(event, snapshot)
        timing_totals["publish"] += perf_counter() - before
        event_index += 1

    def lose(at, reason, *, kind="source_lost", new_health="interrupted"):
        nonlocal needs_reset, health
        session.advance(max(at, session.updated_at), disconnected=True)
        needs_reset = True
        health = new_health
        publish(kind, reason)

    def evaluate(request, image, models, failure):
        return observe_pixels(
            session.setup, request, image, lambda: models, failure=failure
        )

    def heartbeat(reason=None):
        nonlocal health, last_heartbeat
        at = at_now()
        if at < session.updated_at:
            lose(at, "processing_clock_went_backwards")
            return
        if session.last_frame_at is None or needs_reset:
            return
        age = (at - session.last_frame_at).total_seconds()
        if age < heartbeat_after:
            return
        if last_heartbeat is not None and at - last_heartbeat < timedelta(seconds=0.5):
            return
        last_heartbeat = at
        session.advance(at)
        health = "waiting"
        publish("heartbeat", reason)

    worker = ThreadPoolExecutor(max_workers=1) if live else None
    try:
        publish("session_started")
        before_load = perf_counter()
        models = model_loader()
        model_load_seconds = perf_counter() - before_load
        session.check_model(models.provenance)
        session.model_provenance = deepcopy(models.provenance)
        for event in source:
            skipped += event.skipped_frames
            timing_totals["decode"] += event.decode_seconds
            if event.kind == "eof":
                lose(
                    at_now(),
                    event.reason or "end_of_file",
                    kind="eof",
                    new_health="eof",
                )
                break
            if event.kind == "disconnected":
                reason = event.reason or "source_disconnected"
                if not live:
                    lose(at_now(), reason, kind="error", new_health="error")
                    raise RuntimeError(reason)
                lose(at_now(), reason)
                continue
            if event.kind == "heartbeat":
                heartbeat()
                continue
            if event.kind != "frame" or event.frame is None:
                raise ValueError("Invalid source event")
            frame = event.frame
            captured = frame.captured_at
            if frame.camera_id != session.setup.camera_id:
                raise ValueError("Frame camera does not match the approved view")
            # Validate timestamps before constructing replay-compatible observations.
            if (
                not isinstance(captured, datetime)
                or captured.utcoffset() is None
                or type(frame.sequence) is not int
                or frame.sequence < 0
                or (
                    event.pts_seconds is not None
                    and (
                        not isfinite(event.pts_seconds)
                        or event.pts_seconds < 0
                        or (
                            session.latest_pts is not None
                            and event.pts_seconds <= session.latest_pts
                        )
                    )
                )
                or (
                    not needs_reset
                    and session.last_sequence is not None
                    and frame.sequence <= session.last_sequence
                )
                or (
                    session.last_frame_at is not None
                    and captured <= session.last_frame_at
                )
            ):
                discarded += 1
                lose(
                    at_now(),
                    "invalid_or_nonmonotonic_frame_time",
                    kind="frame_rejected",
                )
                continue
            evaluated = clock() if live else captured
            if evaluated < session.updated_at:
                discarded += 1
                lose(
                    session.updated_at,
                    "processing_clock_went_backwards",
                    kind="frame_rejected",
                )
                continue
            if needs_reset:
                session.reset(evaluated)
                needs_reset = False
                health = "recovering"
                publish("source_recovered")

            image = frame.image
            if (
                not isinstance(image, np.ndarray)
                or image.ndim != 3
                or (image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0)
            ):
                image = None
                failure = ("invalid_frame", None)
            else:
                failure = (None, None)
                width, height = session.setup.configuration.camera_image_sizes[
                    frame.camera_id
                ]
                if image.shape[:2] != (height, width):
                    failure = ("dimension_mismatch", None)
                    geometry_valid = False
                elif view_check is not None and not view_check(frame):
                    geometry_valid = False
            session.check_model(models.provenance)
            request = ObservationRequest(
                session_id=session.session_id,
                sequence=frame.sequence,
                captured_at=captured,
                evaluated_at=evaluated,
                max_age_seconds=(
                    session.policy.max_gap_seconds
                    if live
                    else session.policy.max_observation_age_seconds
                ),
                view_confirmed=geometry_valid,
                model=session.backend,
                image_source=source_id,
                image_sha256=(
                    sha256(np.ascontiguousarray(image)).hexdigest()
                    if image is not None
                    else None
                ),
                evaluation_clock="host_utc" if live else "historical_pts",
            )
            before = perf_counter()
            if live:
                future = worker.submit(evaluate, request, image, models, failure)
                while True:
                    try:
                        document = future.result(timeout=1.0)
                        break
                    except TimeoutError:
                        if future.done():
                            document = future.result()
                            break
                        if source_is_current is not None and not source_is_current(
                            event
                        ):
                            if not needs_reset:
                                lose(at_now(), "connection_changed_during_inference")
                        else:
                            heartbeat("inference_in_progress")
                finished_at = clock()
                if needs_reset or (
                    source_is_current is not None and not source_is_current(event)
                ):
                    discarded += 1
                    timing_totals["inference_pipeline"] += perf_counter() - before
                    lose(finished_at, "discarded_observation_from_interrupted_source")
                    continue
                if (
                    finished_at - captured
                ).total_seconds() >= session.policy.max_gap_seconds:
                    document = evaluate(
                        request,
                        image,
                        models,
                        ("stale_frame", "Inference exceeded frame age"),
                    )
                document["evaluated_at"] = iso(finished_at)
            else:
                document = evaluate(request, image, models, failure)
                finished_at = evaluated
            inference_elapsed = perf_counter() - before
            timing_totals["inference_pipeline"] += inference_elapsed
            if finished_at < session.updated_at:
                discarded += 1
                lose(
                    session.updated_at,
                    "processing_clock_went_backwards",
                    kind="frame_rejected",
                )
                continue
            before_temporal = perf_counter()
            session.apply(document, finished_at, event.pts_seconds)
            temporal_elapsed = perf_counter() - before_temporal
            timing_totals["temporal"] += temporal_elapsed
            processed += 1
            health = "streaming" if document["status"] == "observed" else "degraded"
            publish(
                "observation",
                document["status"] if health == "degraded" else None,
                observation=document,
                image_hash_basis="decoded_bgr_uint8",
                source_pts_seconds=event.pts_seconds,
                skipped_since_last_event=event.skipped_frames,
                timing_seconds={
                    "decode": event.decode_seconds,
                    "inference_pipeline": inference_elapsed,
                    "temporal": temporal_elapsed,
                },
            )
        if health != "eof":
            lose(at_now(), "source_finished", kind="stopped", new_health="stopped")
    except KeyboardInterrupt:
        lose(at_now(), "interrupted_by_operator", kind="stopped", new_health="stopped")
    except Exception:
        if health != "error":
            lose(at_now(), "runner_error", kind="error", new_health="error")
        raise
    finally:
        source.close()
        if worker is not None:
            worker.shutdown(wait=False, cancel_futures=True)

    elapsed = perf_counter() - started
    return {
        "processed_frames": processed,
        "skipped_frames": skipped,
        "discarded_frames": discarded,
        "events": event_index,
        "elapsed_seconds": elapsed,
        "observations_per_wall_second": processed / elapsed,
        "stage_totals_seconds": timing_totals,
        "model_load_seconds": model_load_seconds,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--rtsp")
    parser.add_argument("--video-anchor", type=timestamp)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--configuration-revision", type=int)
    parser.add_argument("--parking-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--view-confirmed", required=True, action="store_true")
    parser.add_argument("--diagnostic-slot-id", action="append", default=[])
    parser.add_argument("--model", choices=MODEL_CHOICES, default="acpds-rcnn")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sample-hz", type=positive_seconds, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main():
    from parkvision.capture.occupancy_source import RTSPSource, VideoFileSource

    parser = build_parser()
    args = parser.parse_args()
    if (args.video is None) != (args.video_anchor is None):
        parser.error(
            "--video needs --video-anchor; RTSP uses the host UTC receipt clock"
        )
    try:
        setup = ObservationSetup.read(args.configuration, args.camera_id)
        if (
            args.configuration_sha256 is not None
            and args.configuration_sha256 != setup.configuration_sha256
        ) or (
            args.configuration_revision is not None
            and args.configuration_revision != setup.configuration.revision
        ):
            raise ValueError("Unexpected configuration revision or SHA-256")
        session = LiveSession(
            setup, args.parking_id, args.session_id, args.model, args.diagnostic_slot_id
        )
        if args.video is not None:
            source_id = str(args.video.resolve())
            source = VideoFileSource(
                args.video, args.camera_id, args.video_anchor, sample_hz=args.sample_hz
            )
        else:
            url = urlsplit(args.rtsp)
            if url.scheme != "rtsp" or not url.hostname:
                raise ValueError("Expected an RTSP URL with a host")
            source_id = f"{url.scheme}://{url.hostname}/{args.camera_id}"
            source = RTSPSource(args.rtsp, args.camera_id, sample_hz=args.sample_hz)
        with LiveOutput(args.output_dir) as writer:
            results = run_live(
                session,
                source,
                writer,
                lambda: load_models(args.assets, device=args.device, model=args.model),
                source_id=source_id,
                video_anchor=args.video_anchor,
                view_confirmed=args.view_confirmed,
                sample_hz=args.sample_hz,
            )
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        parser.exit(1, f"Cannot run occupancy: {exc}\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    main()

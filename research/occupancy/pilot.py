"""Capture an operator candidate frame or check a frozen single-camera pilot."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic

import cv2

from apps.dashboard.__main__ import DashboardServer
from apps.dashboard.state import Geometry

from . import live
from .live_output import LiveOutput
from .models import load_models
from parkvision.capture.occupancy_source import RTSP_CLOCK, VIDEO_CLOCK
from parkvision.registration_preview import render_registration_preview

ACTIONS = {
    "authorization": "Potwierdź zgodę właściciela na analizę i lokalny zapis.",
    "source": "Sprawdź źródło, ustawienia kamery i mapowanie CameraId.",
    "configuration": "Zatwierdź fizyczne ROI i sprawdź identyfikatory, rewizję i SHA.",
    "reference": "Sprawdź manifest, obraz, obie maski, akceptację operatora i SHA.",
    "view": "Sprawdź natywny kadr i niezależne pokrycie wszystkich ROI przez maski.",
    "backend": "Sprawdź środowisko research, urządzenie i przypięte zasoby modelu.",
    "output": "Wskaż nowy pusty, zapisywalny katalog wyników.",
    "dashboard": "Sprawdź dostępność portu lokalnego, adapter i tę samą geometrię.",
}


class Blocked(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.details = details


def source_kind(args):
    return "video" if args.video is not None else "rtsp"


@contextmanager
def sample_frame(args):
    source, _ = live.build_source(args)
    deadline = monotonic() + 8  # First attempt: up to 3 s open + 3 s read.
    events = iter(source)
    try:
        for event in events:
            if event.frame is not None:
                yield source, event
                return
            if event.kind != "heartbeat":
                raise Blocked(event.reason or "source_has_no_frame")
            if monotonic() >= deadline:
                break
        raise Blocked("source_frame_timeout")
    finally:
        events.close()


def frame_metadata(event):
    frame = event.frame
    return {
        "image_size": [frame.image.shape[1], frame.image.shape[0]],
        "captured_at": live.iso(frame.captured_at),
        "time_source": event.time_source,
        "pts_seconds": event.pts_seconds,
    }


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG encoding failed")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)


def capture(args):
    if not args.authorized:
        return blocked("authorization", "permission_not_confirmed")
    try:
        with sample_frame(args) as (_, event):
            try:
                save_png(args.image, event.frame.image)
            except (OSError, cv2.error):
                return blocked("output", "cannot_create_candidate_png")
            return {
                "status": "captured",
                "camera_id": args.camera_id,
                "source_kind": source_kind(args),
                **frame_metadata(event),
            }
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        return blocked("source", safe_reason(error, "source"))


def blocked(stage, reason, **details):
    return {
        "status": "blocked",
        "stage": stage,
        "reason": reason,
        "action": ACTIONS[stage],
        **details,
    }


def safe_reason(error, stage):
    # File names and decoder/library error messages can contain private input.
    if isinstance(error, Blocked):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return f"{stage}_file_missing"
    if isinstance(error, PermissionError):
        return f"{stage}_access_denied"
    if isinstance(error, json.JSONDecodeError):
        return f"{stage}_invalid_json"
    message = str(error)
    known = {
        "Unknown camera in configuration:": "camera_id_mismatch",
        "Approved configuration does not match the parking ID": "parking_id_mismatch",
        "Unexpected configuration revision or SHA-256": "configuration_pin_mismatch",
        "approved_": "geometry_approval_required",
        "No configured slot projections": "no_roi_projections",
        "Every configured slot needs at least one projection": "no_roi_projections",
        "At least one physical parking slot": "no_physical_roi",
        "Diagnostic slot IDs": "unknown_diagnostic_roi",
        "Every slot must have a projection": "incomplete_camera_projections",
        "Invalid camera integrity reference manifest": "invalid_reference_manifest",
        "Unsupported camera reference schema": "unsupported_reference_schema",
        "Camera reference requires an approved": "reference_approval_required",
        "Camera reference approval must include": "reference_approval_timezone_missing",
        "Camera reference does not bind": "reference_configuration_mismatch",
        "Camera reference size": "reference_dimensions_mismatch",
        "Camera reference or mask SHA-256 mismatch": "reference_asset_sha_mismatch",
        "Camera reference or mask has invalid pixels": "invalid_reference_or_mask",
        "Static masks": "invalid_static_masks",
        "Static fit and check masks": "overlapping_static_masks",
        "Unexpected camera reference manifest SHA-256": (
            "reference_manifest_sha_mismatch"
        ),
        "CUDA is unavailable": "cuda_unavailable",
        "CUDA detector runtime probe failed": "cuda_detector_probe_failed",
        "Unexpected acpds-rcnn asset hash": "backend_asset_sha_mismatch",
        "Unexpected convnext-roi4 asset hash": "backend_asset_sha_mismatch",
        "Expected an rtsp:// URL": "invalid_rtsp_source",
        "RTSP source missing": "rtsp_secret_missing",
        "--video needs --video-anchor": "source_clock_invalid",
    }
    return next(
        (reason for prefix, reason in known.items() if message.startswith(prefix)),
        f"{stage}_failed",
    )


def check_dashboard(directory, args, session):
    if not 0 <= args.dashboard_port <= 65535:
        raise Blocked("invalid_dashboard_port")
    geometry = Geometry(args.configuration)
    with DashboardServer(args.dashboard_port, directory, geometry) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        worker.start()
        client = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            results = {}
            for route in ("/", "/api/v1/health", "/api/v1/status", "/api/v1/history"):
                client.request("GET", route)
                response = client.getresponse()
                results[route] = response.read()
                if response.status != 200 or not results[route]:
                    raise Blocked("dashboard_route_failed", route=route)
            status = json.loads(results["/api/v1/status"])
            if (
                status["session_id"] != session.session_id
                or not status["geometry"]
                or status["availability_counts"]["confirmed_free"] != 0
            ):
                raise Blocked("dashboard_startup_state_invalid")
            if json.loads(results["/api/v1/health"])["status"] != "ok":
                raise Blocked("dashboard_health_invalid")
            history = json.loads(results["/api/v1/history"])
            if (
                not history["events"]
                or history["events"][0]["event"] != "session_started"
            ):
                raise Blocked("dashboard_history_invalid")
        finally:
            client.close()
            server.shutdown()
            worker.join()


def check_output(directory, session, reference, args):
    at = live.utc_now()
    session.start(at)
    snapshot = session.snapshot(
        source={
            "id": f"preflight/{args.camera_id}",
            "kind": source_kind(args),
            "time_source": VIDEO_CLOCK if args.video is not None else RTSP_CLOCK,
            "health": "starting",
            "reason": None,
            "observations": 0,
            "discarded_frames": 0,
        },
        published_at=at,
        skipped_frames=0,
        view=live.ViewMonitor(reference, 5, 10),
    )
    if (
        snapshot["availability_counts"]["unknown"]
        != snapshot["availability_counts"]["physical_total"]
    ):
        raise Blocked("new_session_not_unknown")
    event = {
        "schema_version": 1,
        "event": "session_started",
        "event_index": 0,
        "snapshot": snapshot,
    }
    with LiveOutput(directory) as output:
        output.publish(event, snapshot)


def prerequisites(args):
    failures = []
    if not args.authorized:
        failures.append(blocked("authorization", "permission_not_confirmed"))
    if args.video is None and args.rtsp is None and not args.rtsp_env:
        failures.append(blocked("source", "source_missing"))
    if not args.view_confirmed:
        failures.append(blocked("source", "source_and_view_not_confirmed"))
    for stage, fields in (
        (
            "configuration",
            (
                "configuration",
                "configuration_sha256",
                "configuration_revision",
                "parking_id",
                "camera_id",
                "session_id",
            ),
        ),
        ("reference", ("view_reference", "view_reference_sha256")),
        ("backend", ("assets",)),
        ("output", ("output_dir",)),
    ):
        missing = [
            "--" + name.replace("_", "-")
            for name in fields
            if getattr(args, name) is None
        ]
        if missing:
            failures.append(blocked(stage, "arguments_missing", arguments=missing))
    return failures


def preflight(args):
    failures = prerequisites(args)
    if failures:
        return {"status": "blocked", "blockers": failures}
    checks = {"authorization": "operator_confirmed", "source_kind": source_kind(args)}
    try:
        stage = "configuration"
        session = live.prepare_session(args)
        setup = session.setup
        checks[stage] = {
            "parking_id": session.setup.configuration.parking.id,
            "camera_id": args.camera_id,
            "revision": setup.configuration.revision,
            "sha256": setup.configuration_sha256,
            "physical_roi_count": len(setup.projections) - len(session.diagnostic),
        }

        stage = "reference"
        reference = live.prepare_reference(args, setup)
        self_check = reference.check(reference.image)
        checks[stage] = asdict(self_check)
        if self_check.status != "verified":
            raise Blocked(self_check.reason, integrity_status=self_check.status)

        stage = "source"
        is_file = args.video is not None
        with sample_frame(args) as (source, event):
            checks[stage] = frame_metadata(event)
            stage = "view"
            if checks["source"]["image_size"] != list(reference.image_size):
                raise Blocked(
                    "resolution_mismatch",
                    expected_image_size=list(reference.image_size),
                    integrity_status="mismatch",
                )

            def review(registration):
                preview = render_registration_preview(
                    reference.image, event.frame.image, asdict(registration)
                )
                try:
                    save_png(args.alignment_preview, preview)
                except (OSError, cv2.error) as error:
                    raise Blocked("cannot_create_alignment_preview") from error

            result = reference.check(
                event.frame.image, review=review if args.alignment_preview else None
            )
            checks[stage] = asdict(result)
            if result.status != "verified":
                raise Blocked(result.reason, integrity_status=result.status)
            if not is_file and not source.is_current(event):
                raise Blocked("source_interrupted_during_view_check")

        stage = "backend"
        models = load_models(args.assets, device=args.device, model=args.model)
        session.check_model(models.provenance)
        selection = models.provenance["device_selection"]
        checks[stage] = {
            "model": args.model,
            "checkpoint_sha256": session.checkpoint,
            "requested_device": args.device,
            "selected_device": selection["selected"],
            "device_reason": selection["reason"].split(":", 1)[0],
        }
        del models

        stage = "output"
        args.output_dir.mkdir(mode=0o700, exist_ok=True)
        if any(args.output_dir.iterdir()):
            raise Blocked("output_directory_not_empty")
        with TemporaryDirectory(prefix=".preflight-", dir=args.output_dir) as temp:
            directory = Path(temp)
            check_output(directory, session, reference, args)
            checks[stage] = "new_session_written_with_unknown_slots"
            stage = "dashboard"
            check_dashboard(directory, args, session)
            checks[stage] = "loopback_http_verified"
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        HTTPException,
        cv2.error,
    ) as e:
        details = e.details if isinstance(e, Blocked) else {}
        return {
            **blocked(stage, safe_reason(e, stage), **details),
            "checks": checks,
        }

    conditions = []
    if is_file:
        conditions.append("file_only_rtsp_not_checked")
    if selection["selected"] == "cpu":
        conditions.append("cpu_only_cuda_pilot_not_checked")
    return {
        "status": "conditional" if conditions else "ready",
        "conditions": conditions,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    capture_parser = commands.add_parser("capture", help="Save one candidate PNG")
    capture_parser.set_defaults(command=capture)
    live.add_source_arguments(capture_parser)
    capture_parser.add_argument("--authorized", action="store_true")
    capture_parser.add_argument("--camera-id", required=True)
    capture_parser.add_argument("--image", type=Path, required=True)

    preflight_parser = commands.add_parser("preflight", help="Check pilot readiness")
    preflight_parser.set_defaults(command=preflight)
    live.add_source_arguments(preflight_parser, required=False)
    live.add_setup_arguments(preflight_parser, required=False)
    preflight_parser.add_argument("--authorized", action="store_true")
    preflight_parser.add_argument("--dashboard-port", type=int, default=8765)
    preflight_parser.add_argument("--alignment-preview", type=Path)

    args = parser.parse_args()
    result = args.command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit({"blocked": 2, "conditional": 3}.get(result["status"], 0))


if __name__ == "__main__":
    main()

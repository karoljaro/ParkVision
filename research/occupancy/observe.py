"""Observe saved parking ROIs with an explicitly selected local occupancy model."""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

# Running this research script needs the owning engine's existing domain models.
ENGINE_SOURCE = Path(__file__).resolve().parents[2] / "apps/engine/src"
if ENGINE_SOURCE.is_dir():
    sys.path.insert(0, str(ENGINE_SOURCE))
# Support both direct execution from any cwd and package imports. The ConvNeXt
# adapter imports the existing training implementation by package name.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkvision.capture.frame import Frame  # noqa: E402
from parkvision.configuration import load_configuration  # noqa: E402
from parkvision.domain.types import CameraId  # noqa: E402

from research.occupancy.models import MODEL_CHOICES, load_models  # noqa: E402
from research.occupancy.rules import (  # noqa: E402
    BASELINE_ID,
    assign_vehicles,
    classify,
)


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an ISO 8601 timestamp") from error
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError("Timestamp must include its UTC offset")
    return result.astimezone(timezone.utc)


def positive_seconds(value):
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected a positive number of seconds"
        ) from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Expected a positive number of seconds")
    return result


def sequence_number(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an integer sequence") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Sequence must be nonnegative")
    return number


def input_reason(
    image,
    camera_id,
    configuration,
    captured_at,
    evaluated_at,
    max_age_seconds,
    view_confirmed,
):
    if image is None:
        return "invalid_frame"
    h, w = image.shape[:2]
    if (w, h) != configuration.camera_image_sizes[camera_id]:
        return "dimension_mismatch"
    configuration.projections_for_camera(camera_id, image_width=w, image_height=h)
    if not view_confirmed:
        return "view_unverified"
    if captured_at is None:
        return "missing_capture_time"
    age = (evaluated_at - captured_at).total_seconds()
    if age < 0:
        return "future_frame"
    if age > max_age_seconds:
        return "stale_frame"
    return None


def observe(args, models_factory=None):
    selected_model = getattr(args, "model", "acpds-rcnn")
    if selected_model not in MODEL_CHOICES:
        raise ValueError(f"Unknown occupancy model: {selected_model}")
    config_bytes = args.configuration.read_bytes()
    configuration = load_configuration(args.configuration)
    if args.configuration.read_bytes() != config_bytes:
        raise ValueError("Configuration changed during reading")
    if not args.session_id.strip():
        raise ValueError("Session ID must be nonempty")
    if args.camera_id not in configuration.camera_image_sizes:
        raise ValueError(f"Unknown camera in configuration: {args.camera_id}")
    width, height = configuration.camera_image_sizes[args.camera_id]
    projections = configuration.projections_for_camera(
        args.camera_id, image_width=width, image_height=height
    )
    if not projections:
        raise ValueError("No configured slot projections for this camera")
    unreadable = set(args.unreadable_slot_id)
    slot_ids = {p.parking_slot_id for p in projections}
    if not unreadable <= slot_ids:
        raise ValueError("Unreadable slot IDs must belong to the selected camera")
    evaluated_at = args.evaluated_at or datetime.now(timezone.utc)
    config_sha = sha256(config_bytes).hexdigest()
    raw = None
    image = None
    reason = None
    detail = None
    try:
        raw = args.image.read_bytes()
        if raw:
            image = cv2.imdecode(
                np.frombuffer(raw, np.uint8),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
    except FileNotFoundError:
        reason = "missing_frame"
    except OSError as error:
        reason, detail = "invalid_frame", str(error)
    except cv2.error as error:
        reason, detail = "invalid_frame", str(error)
    if reason is None:
        reason = input_reason(
            image,
            args.camera_id,
            configuration,
            args.captured_at,
            evaluated_at,
            args.max_age_seconds,
            args.view_confirmed,
        )
    image_sha = sha256(raw).hexdigest() if raw is not None else None
    provenance = {
        "baseline_id": BASELINE_ID,
        "models": None,
        "selected_model": selected_model,
    }
    observations = [
        {
            "slot_id": projection.parking_slot_id,
            "state": "unknown",
            "unknown_reason": reason,
            "confidence": None,
            "p_occupied": None,
            "detector_D": "unknown",
            "detection_ids": [],
            "visibility": (
                "operator_unreadable"
                if projection.parking_slot_id in unreadable
                else "unassessed"
            ),
        }
        for projection in projections
    ]
    detections = []
    if reason is None:
        frame = Frame(
            camera_id=str(args.camera_id),
            sequence=args.sequence,
            captured_at=args.captured_at,
            image=image,
        )
        try:
            models = (
                models_factory(args.assets, device=args.device)
                if models_factory is not None
                else load_models(args.assets, device=args.device, model=selected_model)
            )
            provenance["models"] = models.provenance
            probabilities, detections = models.predict(frame, projections)
            if len(probabilities) != len(projections):
                raise ValueError(
                    "Per-slot output does not cover exactly the camera ROIs"
                )
            assignments = assign_vehicles(projections, width, height, detections)
            # Build all results before publishing so a late failure cannot leave
            # a partially trusted frame with mixed error and inferred states.
            inferred = []
            for entry, probability in zip(observations, probabilities, strict=True):
                ids = assignments[entry["slot_id"]]
                state, unknown_reason = classify(probability, ids)
                if entry["slot_id"] in unreadable:
                    state, unknown_reason = "unknown", "occluded"
                inferred.append(
                    {
                        **entry,
                        "state": state,
                        "unknown_reason": unknown_reason,
                        "p_occupied": probability,
                        "detector_D": "occupied" if ids else "unknown",
                        "detection_ids": ids,
                    }
                )
            observations = inferred
        except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
            reason, detail = "inference_error", str(error)
            observations = [
                {**entry, "unknown_reason": reason} for entry in observations
            ]
            detections = []

    return {
        "schema_version": 1,
        "kind": "experimental_offline_occupancy",
        "parking_id": configuration.parking.id,
        "configuration_revision": configuration.revision,
        "configuration_sha256": config_sha,
        "camera_id": args.camera_id,
        "source_session": args.session_id,
        "sequence": args.sequence,
        "captured_at": args.captured_at.isoformat() if args.captured_at else None,
        "evaluated_at": evaluated_at.isoformat(),
        "evaluation_clock": "provided" if args.evaluated_at else "system_utc",
        "max_age_seconds": args.max_age_seconds,
        "view_confirmed": args.view_confirmed,
        "image": {
            "path": str(args.image.resolve()),
            "sha256": image_sha,
            "width": int(image.shape[1]) if image is not None else None,
            "height": int(image.shape[0]) if image is not None else None,
        },
        "status": reason or "observed",
        "error_detail": detail,
        "provenance": provenance,
        "observations": observations,
        "detections": detections,
    }


def save_result(path, result):
    """Publish a complete JSON exclusively, without replacing existing files."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Linking is exclusive even when another writer creates path concurrently.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--camera-id", type=CameraId, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--captured-at", type=timestamp)
    parser.add_argument("--evaluated-at", type=timestamp)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sequence", type=sequence_number, default=0)
    parser.add_argument("--max-age-seconds", type=positive_seconds, required=True)
    parser.add_argument("--view-confirmed", action="store_true")
    parser.add_argument("--unreadable-slot-id", action="append", default=[])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="acpds-rcnn")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Output already exists: {args.output}")
        result = observe(args)
        save_result(args.output, result)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot observe frame: {error}\n")
    print(f"Saved {len(result['observations'])} observations to {args.output}")
    if result["status"] == "inference_error":
        parser.exit(1, f"Inference failed: {result['error_detail']}\n")


if __name__ == "__main__":
    main()

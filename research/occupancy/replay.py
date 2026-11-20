"""Deterministically replay saved offline observations for one camera/slot."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ENGINE_SOURCE = Path(__file__).resolve().parents[2] / "apps/engine/src"
if ENGINE_SOURCE.is_dir():
    sys.path.insert(0, str(ENGINE_SOURCE))

from parkvision.domain.temporal_occupancy import (  # noqa: E402
    SlotObservation,
    StreamKey,
    TemporalPolicy,
    disconnect,
    initial_state,
    observe_slot,
    reset_stream,
    tick,
)

# Reuse the offline command's exclusive, atomic publication; no model is loaded.
from models import CONVNEXT_SHA256, SLOT_SHA256  # noqa: E402
from observe import save_result  # noqa: E402

AGGREGATOR_ID = "temporal-occupancy-v1"


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Expected an ISO 8601 timestamp with UTC offset")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("Timestamp must include its UTC offset")
    return result.astimezone(timezone.utc)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    raw = path.read_bytes()
    return (
        json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=reject_constant
        ),
        sha256(raw).hexdigest(),
    )


def observation_from_document(
    data, slot_id, source_id, *, motion="unassessed", allow_missing=False
):
    """Adapt schema v1 without reclassifying scores or changing the raw file."""
    if (
        type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "experimental_offline_occupancy"
    ):
        raise ValueError("Expected an offline occupancy schema v1 document")
    key = StreamKey(
        parking_id=data["parking_id"],
        camera_id=data["camera_id"],
        slot_id=slot_id,
        configuration_revision=data["configuration_revision"],
        configuration_sha256=data["configuration_sha256"],
        session_id=data["source_session"],
    )
    rows = data["observations"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("observations must be an array of objects")
    ids = [row["slot_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate slot observation")
    matches = [row for row in rows if row["slot_id"] == slot_id]
    if not matches and not allow_missing:
        raise ValueError("First observation must contain the selected slot_id")
    row = (
        matches[0]
        if matches
        else {
            "state": "unknown",
            "unknown_reason": "missing_slot_observation",
            "visibility": "unassessed",
        }
    )
    return SlotObservation(
        key=key,
        sequence=data["sequence"],
        captured_at=(timestamp(data["captured_at"]) if data["captured_at"] else None),
        evaluated_at=timestamp(data["evaluated_at"]),
        state=row["state"],
        unknown_reason=row["unknown_reason"],
        source_id=source_id,
        image_sha256=data["image"]["sha256"],
        view_confirmed=data["view_confirmed"],
        frame_status=data["status"],
        visibility=row["visibility"],
        motion=motion,
    )


def json_value(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def observation_model_identity(data):
    provenance = data.get("provenance") or {}
    models = provenance.get("models") or {}
    per_slot = models.get("per_slot") or {}
    checkpoint = per_slot.get("sha256")
    backend = provenance.get("selected_model")
    # Older schema v1 runs identified the loaded checkpoint instead of the CLI model.
    known_backend = {
        SLOT_SHA256: "acpds-rcnn",
        CONVNEXT_SHA256: "convnext-roi4",
    }.get(checkpoint)
    if backend is not None and known_backend is not None and backend != known_backend:
        raise ValueError("Occupancy backend does not match its checkpoint")
    return {
        field: value
        for field, value in {
            "backend": backend if backend is not None else known_backend,
            "per_slot_checkpoint": checkpoint,
        }.items()
        if value is not None
    }


def replay(manifest_path, slot_id, policy=None):
    """Input order is arrival order. No sorting, wall clock, weights or inference."""
    manifest, manifest_sha = read_json(manifest_path)
    if set(manifest) - {"schema_version", "policy", "events", "description"}:
        raise ValueError("Unsupported manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected replay manifest schema_version=1")
    if policy is not None and "policy" in manifest:
        raise ValueError("Specify policy either in the manifest or as an argument")
    policy = policy or TemporalPolicy(**manifest.get("policy", {}))
    events = manifest["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("Replay requires at least one observation event")
    state = None
    stream_model = {}
    snapshots = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("Every replay event must be an object")
        source = None
        source_sha = None
        raw_observation = None
        if "observation" in event:
            if set(event) - {"observation", "at", "motion", "reset"}:
                raise ValueError("Unsupported observation event fields")
            source = event["observation"]
            data, source_sha = read_json(manifest_path.parent / source)
            identity = observation_model_identity(data)
            raw_observation = observation_from_document(
                data,
                slot_id,
                source_sha,
                motion=event.get("motion", "unassessed"),
                allow_missing=state is not None and not event.get("reset", False),
            )
            at = (
                timestamp(event["at"])
                if "at" in event
                else (raw_observation.evaluated_at)
            )
            if type(event.get("reset", False)) is not bool:
                raise ValueError("reset must be a boolean")
            if state is None:
                if event.get("reset", False):
                    raise ValueError("Cannot reset before the first observation")
                state = initial_state(raw_observation.key, at)
            elif event.get("reset", False):
                state = reset_stream(state, raw_observation.key, at)
                stream_model.clear()
            if any(
                key in stream_model and stream_model[key] != value
                for key, value in identity.items()
            ):
                raise ValueError("Mixed occupancy model; use a separate replay stream")
            stream_model.update(identity)
            update = observe_slot(state, raw_observation, at, policy)
            state, disposition = update.state, update.disposition
            kind = "observation"
        else:
            if state is None:
                raise ValueError("Replay must begin with an observation")
            if set(event) == {"tick_at"}:
                state = tick(state, timestamp(event["tick_at"]), policy)
                kind = "tick"
            elif set(event) == {"disconnect_at"}:
                state = disconnect(state, timestamp(event["disconnect_at"]))
                kind = "disconnect"
            else:
                raise ValueError("Expected observation, tick_at or disconnect_at")
            disposition = "accepted"
        snapshots.append(
            {
                "event_index": index,
                "event_type": kind,
                "source": source,
                "source_sha256": source_sha,
                "raw_observation": (
                    asdict(raw_observation) if raw_observation is not None else None
                ),
                "disposition": disposition,
                "temporal": asdict(state),
            }
        )
    return json_value(
        {
            "schema_version": 1,
            "kind": "experimental_temporal_occupancy",
            "aggregator_id": AGGREGATOR_ID,
            "manifest_sha256": manifest_sha,
            "policy": asdict(policy),
            "clock": "manifest_or_offline_evaluated_at",
            "snapshots": snapshots,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Output already exists: {args.output}")
        result = replay(args.manifest, args.slot_id)
        save_result(args.output, result)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Cannot replay observations: {error}\n")
    print(f"Saved {len(result['snapshots'])} temporal snapshots to {args.output}")


if __name__ == "__main__":
    main()

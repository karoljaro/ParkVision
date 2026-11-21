"""Validate the runner's output and project a dashboard-safe read model."""

import json
import re
from datetime import datetime, timedelta, timezone
from hashlib import sha256

SCHEMA_VERSION = 1
SNAPSHOT_MAX_BYTES = 2 * 1024 * 1024
HISTORY_MAX_BYTES = 4 * 1024 * 1024
HISTORY_LIMIT = 50
PUBLICATION_TTL = timedelta(seconds=5)

SOURCE_STATES = (
    "starting",
    "streaming",
    "waiting",
    "recovering",
    "degraded",
    "interrupted",
    "eof",
    "stopped",
    "error",
)
VIEW_STATES = ("checking", "verified", "mismatch", "unverifiable")
ESTIMATES = ("free", "occupied", "unknown")
TEMPORAL_STATES = ("confirmed", "held", "unknown", "stale")
COUNT_KEYS = ("confirmed_free", "confirmed_occupied", "held", "unknown", "stale")
TERMINAL = ("interrupted", "eof", "stopped", "error")
EVENT_TYPES = (
    "session_started",
    "observation",
    "camera_integrity",
    "view_expired",
    "heartbeat",
    "source_lost",
    "source_recovered",
    "frame_rejected",
    "eof",
    "stopped",
    "error",
)


class StateError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.code = code
        self.status = status


def read_json(data):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate keys")
        return result

    def finite(_):
        raise ValueError("non-finite JSON value")

    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=finite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise StateError("invalid_json") from exc


def stamp(value, *, nullable=False):
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ):
        raise StateError("invalid_timestamp")
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise StateError("invalid_timestamp") from exc


def utc(value):
    return value.isoformat(timespec="microseconds") if value is not None else None


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.:-]{1,160}", value):
        raise StateError("invalid_snapshot")
    return value


def reason(value):
    # Error messages and payload details can contain paths, credentials or URLs.
    if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value):
        return value
    return None


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise StateError("invalid_snapshot")
    return value


def integer(value, *, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not int or not 0 <= value <= (2**53 - 1):
        raise StateError("invalid_snapshot")
    return value


def enum(value, expected, *, nullable=False):
    if value is None and nullable:
        return None
    if value not in expected:
        raise StateError("invalid_snapshot")
    return value


def flag(value):
    if type(value) is not bool:
        raise StateError("invalid_snapshot")
    return value


def number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not 0 <= value <= 10**12:
        raise StateError("invalid_snapshot")
    return value


def snapshot(doc):
    """Read only fields used by the dashboard. Never return arbitrary nested data."""
    if not isinstance(doc, dict) or (
        type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
        or doc.get("kind") != "parkvision_live_occupancy"
    ):
        raise StateError("unsupported_snapshot_schema")
    if "camera_integrity" not in doc:
        raise StateError("unsupported_legacy_snapshot")
    try:
        return _snapshot(doc)
    except (KeyError, TypeError) as exc:
        raise StateError("invalid_snapshot") from exc


def _snapshot(doc):
    run = identifier(doc["run_id"])
    session = doc["session_id"]
    if not isinstance(session, str) or len(session) > 240:
        raise StateError("invalid_snapshot")
    prefix, run_part, generation = (
        session.rsplit("/", 2) if session.count("/") >= 2 else ("", "", "")
    )
    if (
        not all(re.fullmatch(r"[\w .:-]{1,120}", part) for part in prefix.split("/"))
        or run_part != run
        or not re.fullmatch(r"\d{1,12}", generation)
    ):
        raise StateError("invalid_snapshot")

    config, source, view, model = (
        doc[key] for key in ("configuration", "source", "camera_integrity", "model")
    )
    kind = enum(source["kind"], ("rtsp", "video"))
    health = enum(source["health"], SOURCE_STATES)
    published = stamp(doc["published_at"])
    updated = stamp(doc["updated_at"])
    captured = stamp(doc["captured_at"], nullable=True)
    checked = stamp(view["checked_at"], nullable=True)
    view_until = stamp(view["valid_until"], nullable=True)
    view_status = enum(view["status"], VIEW_STATES)
    if (
        (kind == "rtsp" and updated > published)
        or (captured is not None and captured > updated)
        or (checked is not None and checked > updated)
        or (
            view_status == "verified"
            and (checked is None or view_until is None or view_until <= checked)
        )
    ):
        raise StateError("invalid_timestamp")

    if not isinstance(doc["slots"], list) or not doc["slots"]:
        raise StateError("invalid_snapshot")
    ids = set()
    slots = []
    for row in doc["slots"]:
        slot_id = identifier(row["slot_id"])
        if slot_id in ids:
            raise StateError("invalid_snapshot")
        ids.add(slot_id)
        temporal = enum(row["temporal_status"], TEMPORAL_STATES)
        estimated = enum(row["estimated_state"], ESTIMATES)
        available = flag(row["confirmed_available"])
        diagnostic = flag(row["diagnostic_only"])
        expiry = stamp(row["confirmed_evidence_expires_at"], nullable=True)
        if (
            (temporal in ("confirmed", "held")) != (estimated != "unknown")
            or (temporal == "confirmed" and (expiry is None or captured is None))
            or (
                available
                and (diagnostic or temporal != "confirmed" or estimated != "free")
            )
        ):
            raise StateError("invalid_snapshot")
        if not diagnostic:
            slots.append(
                {
                    "slot_id": slot_id,
                    "raw_state": enum(row["raw_state"], ESTIMATES, nullable=True),
                    "unknown_reason": reason(row["unknown_reason"]),
                    "estimated_state": estimated,
                    "temporal_status": temporal,
                    "temporal_reason": reason(row["temporal_reason"]),
                    "last_evidence_age_seconds": number(
                        row["last_evidence_age_seconds"]
                    ),
                    "confirmed_evidence_expires_at": utc(expiry),
                    "confirmed_available": available,
                }
            )
    if not slots or integer(config["revision"]) == 0:
        raise StateError("invalid_snapshot")

    selected = model["device"]["selected"]
    if selected is not None and (
        not isinstance(selected, str) or not re.fullmatch(r"cpu|cuda:\d+", selected)
    ):
        raise StateError("invalid_snapshot")
    return {
        "parking_id": identifier(doc["parking_id"]),
        "camera_id": identifier(doc["camera_id"]),
        "run_id": run,
        "session_id": session,
        "configuration": {
            "revision": config["revision"],
            "sha256": digest(config["sha256"]),
        },
        "model": {
            "backend": enum(model["backend"], ("acpds-rcnn", "convnext-roi4")),
            "checkpoint_sha256": digest(model["checkpoint_sha256"]),
            "device": selected,
        },
        "source": {"kind": kind, "health": health, "reason": reason(source["reason"])},
        "camera_integrity": {
            "status": view_status,
            "reason": reason(view["reason"]),
            "checked_at": utc(checked),
            "valid_until": utc(view_until),
            "check_in_progress": flag(view["check_in_progress"]),
        },
        "published_at": utc(published),
        "updated_at": utc(updated),
        "captured_at": utc(captured),
        "sequence": integer(doc["sequence"], nullable=True),
        "source_pts_seconds": number(doc["source_pts_seconds"]),
        "skipped_frames": integer(doc["skipped_frames"]),
        "slots": slots,
    }


def current_status(data, now):
    """Apply availability gates to existing estimates without a temporal replay."""
    source, view = data["source"], data["camera_integrity"]
    live = source["kind"] == "rtsp"
    as_of = now if live else stamp(data["updated_at"])
    published = stamp(data["published_at"])
    if published > now:
        raise StateError("future_publication")
    publication_until = published + PUBLICATION_TTL if live else None
    publication_expired = live and now >= publication_until
    view_until = stamp(view["valid_until"], nullable=True)
    view_expired = view["status"] == "verified" and as_of >= view_until
    if publication_expired:
        freshness = "stale"
    elif source["health"] in TERMINAL:
        freshness = "offline"
    elif view_expired:
        freshness = "stale"
    elif view["status"] != "verified" or source["health"] != "streaming":
        freshness = "unverified"
    else:
        freshness = "current"
    counts = dict.fromkeys(COUNT_KEYS, 0)
    slots = []
    for row in data["slots"]:
        temporal, estimated = row["temporal_status"], row["estimated_state"]
        category = temporal
        effective_until = None
        if temporal in ("confirmed", "held"):
            deadline = stamp(row["confirmed_evidence_expires_at"], nullable=True)
            if (
                publication_expired
                or view_expired
                or (temporal == "confirmed" and as_of >= deadline)
            ):
                category = "stale"
            elif view["status"] != "verified" or source["health"] in TERMINAL:
                category = "unknown"
            elif temporal == "confirmed":
                if source["health"] != "streaming" or (
                    estimated == "free" and not row["confirmed_available"]
                ):
                    category = "unknown"
                else:
                    category = f"confirmed_{estimated}"
                    if live:
                        effective_until = min(publication_until, view_until, deadline)
            elif live:
                effective_until = min(publication_until, view_until)
        counts[category] += 1
        slots.append(
            {
                **row,
                "effective_status": category,
                "effective_until": utc(effective_until),
                "confirmed_available": category == "confirmed_free",
            }
        )
    return {
        **data,
        "schema_version": SCHEMA_VERSION,
        "mode": "live" if live else "demo",
        "server_time": utc(now),
        "freshness": freshness,
        "publication_valid_until": utc(publication_until),
        "availability_counts": {"physical_total": len(slots), **counts},
        "slots": slots,
    }


def read_snapshot(path):
    try:
        with path.open("rb") as file:
            raw = file.read(SNAPSHOT_MAX_BYTES + 1)
    except FileNotFoundError as exc:
        raise StateError("snapshot_missing", 404) from exc
    except OSError as exc:
        raise StateError("snapshot_unavailable", 503) from exc
    if len(raw) > SNAPSHOT_MAX_BYTES:
        raise StateError("snapshot_too_large")
    return snapshot(read_json(raw))


def _journal_tail(path, count):
    """Read a bounded suffix and return completed, newline-terminated records."""
    try:
        with path.open("rb") as file:
            end = file.seek(0, 2)
            start = end
            blocks = []
            lines = 0
            while start and end - start < HISTORY_MAX_BYTES and lines <= count:
                size = min(65536, start, HISTORY_MAX_BYTES - (end - start))
                start -= size
                file.seek(start)
                block = file.read(size)
                blocks.append(block)
                lines += block.count(b"\n")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise StateError("journal_unavailable", 503) from exc
    records = b"".join(reversed(blocks)).split(b"\n")
    # A partial tail is uncommitted. With a nonzero seek the first line may be cut.
    return records[bool(start) : -1][-count:]


def _event(record):
    doc = read_json(record)
    if (
        not isinstance(doc, dict)
        or type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
    ):
        raise StateError("unsupported_journal_schema")
    try:
        data = snapshot(doc["snapshot"])
        event = enum(doc["event"], EVENT_TYPES)
        index = integer(doc["event_index"])
    except (KeyError, TypeError) as exc:
        raise StateError("invalid_journal") from exc
    return data, event, index


def read_history(path, limit):
    result = []
    previous = None
    for record in _journal_tail(path, limit + 1):
        data, event, index = _event(record)
        same_session = (
            previous is not None and previous["session_id"] == data["session_id"]
        )
        old_rows = {r["slot_id"]: r for r in previous["slots"]} if same_session else {}
        rows = []
        for row in data["slots"]:
            old = old_rows.get(row["slot_id"])
            changed = old is not None and any(
                old[key] != row[key]
                for key in ("estimated_state", "temporal_status", "unknown_reason")
            )
            if changed or event == "observation":
                rows.append(
                    {
                        "slot_id": row["slot_id"],
                        "previous_state": old["estimated_state"] if old else None,
                        "previous_temporal_status": (
                            old["temporal_status"] if old else None
                        ),
                        "state": row["estimated_state"],
                        "temporal_status": row["temporal_status"],
                        "unknown_reason": row["unknown_reason"],
                        "changed": changed,
                    }
                )
        view, source = data["camera_integrity"], data["source"]
        result.append(
            {
                "event": event,
                "event_index": index,
                "run_id": data["run_id"],
                "session_id": data["session_id"],
                "published_at": data["published_at"],
                "observed_at": data["updated_at"],
                "source_kind": source["kind"],
                "source_health": source["health"],
                "source_reason": source["reason"],
                "camera_status": view["status"],
                "camera_reason": view["reason"],
                "session_reset": (
                    event == "source_recovered"
                    or (previous is not None and not same_session)
                ),
                "slots": rows,
            }
        )
        previous = data
    return {"schema_version": SCHEMA_VERSION, "events": result[-limit:]}


class Geometry:
    """Optional approved projection source pinned at startup."""

    def __init__(self, path):
        from parkvision.configuration import load_configuration

        self.sha256 = sha256(path.read_bytes()).hexdigest()
        self.configuration = load_configuration(path)
        if (
            sha256(path.read_bytes()).hexdigest() != self.sha256
            or not self.configuration.is_approved
        ):
            raise ValueError("Geometry changed during loading or lacks approval")

    def for_snapshot(self, data):
        if (
            self.sha256 != data["configuration"]["sha256"]
            or self.configuration.revision != data["configuration"]["revision"]
            or self.configuration.parking.id != data["parking_id"]
            or data["camera_id"] not in self.configuration.camera_image_sizes
        ):
            return None
        width, height = self.configuration.camera_image_sizes[data["camera_id"]]
        projections = {
            p.parking_slot_id: p
            for p in self.configuration.projections_for_camera(
                data["camera_id"], image_width=width, image_height=height
            )
        }
        slots = [row["slot_id"] for row in data["slots"]]
        if not set(slots) <= projections.keys():
            return None
        return {
            "image_size": [width, height],
            "projections": [
                {
                    "slot_id": slot_id,
                    "vertices": [
                        [point.x, point.y]
                        for point in projections[slot_id].polygon.vertices
                    ],
                }
                for slot_id in slots
            ],
        }

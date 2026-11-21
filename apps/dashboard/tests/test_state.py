"""Contract regressions drawn from the Stage 2 runner output shape."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from apps.dashboard.__main__ import DashboardServer
from apps.dashboard.state import (
    Geometry,
    StateError,
    current_status,
    read_history,
    read_snapshot,
    snapshot,
)
from parkvision.configuration import load_configuration, save_configuration
from parkvision.domain.geometry import Point, Polygon
from research.occupancy.live_output import LiveOutput

NOW = datetime(2026, 11, 21, 19, 30, tzinfo=timezone.utc)
CONFIGURATION = Path("apps/engine/examples/parking-setup-v1.json")
CONFIG_SHA = "7ecf5c8828c1800b212457bdd70d99421c4c297339af340a01c48f15e013becc"


def at(seconds):
    return (NOW + timedelta(seconds=seconds)).isoformat()


def slot(name="slot-01", temporal="confirmed", state="free", diagnostic=False):
    return {
        "slot_id": name,
        "diagnostic_only": diagnostic,
        "raw_state": state,
        "estimated_state": state,
        "unknown_reason": "view_unverified" if state == "unknown" else None,
        "temporal_status": temporal,
        "temporal_reason": "repeated_evidence",
        "last_evidence_age_seconds": 0.1,
        "confirmed_evidence_expires_at": at(3) if temporal == "confirmed" else None,
        "confirmed_available": not diagnostic
        and state == "free"
        and temporal == "confirmed",
    }


def example(kind="rtsp"):
    return {
        "schema_version": 1,
        "kind": "parkvision_live_occupancy",
        "parking_id": "synthetic-parking",
        "camera_id": "camera-east",
        "run_id": "run-1",
        "session_id": "test/run-1/0",
        "configuration": {"revision": 1, "sha256": CONFIG_SHA},
        "model": {
            "backend": "acpds-rcnn",
            "checkpoint_sha256": "a" * 64,
            "device": {"selected": "cuda:0", "reason": "/secret/env"},
            "provenance": {"private": "/secret/weights"},
        },
        "source": {
            "kind": kind,
            "health": "streaming",
            "id": "rtsp://secret@camera",
            "reason": "/private/error",
        },
        "camera_integrity": {
            "status": "verified",
            "reason": "native_roi_alignment_verified",
            "check_in_progress": True,
            "checked_at": at(-1),
            "valid_until": at(3.5),
            "evidence": {"private": "/secret/reference"},
        },
        "published_at": at(-0.1),
        "updated_at": at(-0.2),
        "captured_at": at(-0.3),
        "sequence": 9,
        "source_pts_seconds": 9.0 if kind == "video" else None,
        "skipped_frames": 13,
        "availability_counts": {"confirmed_free": 42},
        "slots": [slot(), slot("slot-02", diagnostic=True)],
    }


def test_live_uses_engine_states_and_deadlines_and_omits_private_fields():
    data = current_status(snapshot(example()), NOW)
    assert data["mode"] == "live"
    assert data["slots"][0]["effective_status"] == "confirmed_free"
    assert data["slots"][0]["effective_until"] == at(3).replace("+", ".000000+")
    assert data["camera_integrity"]["check_in_progress"]
    assert data["availability_counts"]["confirmed_free"] == 1
    assert data["availability_counts"]["physical_total"] == 1
    assert data["source"]["reason"] is None
    assert "secret" not in json.dumps(data)


@pytest.mark.parametrize(
    "content,err",
    [
        (b"not json", "invalid_json"),
        (b'{"a":1,"a":2}', "invalid_json"),
        (b"42", "unsupported_snapshot_schema"),
        (b'{"schema_version":2}', "unsupported_snapshot_schema"),
    ],
)
def test_invalid_file_reports_a_safe_code(tmp_path, content, err):
    filename = tmp_path / "snapshot.json"
    with pytest.raises(StateError, match="snapshot_missing") as missing:
        read_snapshot(filename)
    assert missing.value.status == 404
    filename.write_bytes(content)
    with pytest.raises(StateError, match=err) as invalid:
        read_snapshot(filename)
    assert invalid.value.status == 422


def test_legacy_stage_one_snapshot_cannot_supply_camera_proof():
    doc = example()
    del doc["camera_integrity"]
    with pytest.raises(StateError, match="unsupported_legacy_snapshot"):
        snapshot(doc)


@pytest.mark.parametrize("value", [None, "2026-11-21", "2026-11-21T19:30:00", "bad"])
def test_rejects_incomplete_timestamp(value):
    doc = example()
    doc["published_at"] = value
    with pytest.raises(StateError, match="invalid_timestamp"):
        snapshot(doc)


def test_rejects_missing_confirmed_deadline_and_future_publication():
    doc = example()
    doc["slots"][0]["confirmed_evidence_expires_at"] = None
    with pytest.raises(StateError, match="invalid_snapshot"):
        snapshot(doc)
    doc = example()
    doc["published_at"] = at(1)
    with pytest.raises(StateError, match="future_publication"):
        current_status(snapshot(doc), NOW)


@pytest.mark.parametrize(
    "seconds,view_until,evidence_until,freshness",
    [
        (5, 10, 10, "stale"),
        (3.5, 3.5, 4.5, "stale"),
        (3, 3.5, 3, "current"),
    ],
)
def test_expires_exactly_at_each_deadline(
    seconds, view_until, evidence_until, freshness
):
    doc = example()
    doc["camera_integrity"]["valid_until"] = at(view_until)
    doc["slots"][0]["confirmed_evidence_expires_at"] = at(evidence_until)
    data = current_status(snapshot(doc), NOW + timedelta(seconds=seconds))
    assert data["freshness"] == freshness
    assert data["availability_counts"]["stale"] == 1
    assert not data["slots"][0]["confirmed_available"]


def test_current_held_unknown_stale_and_occupied_partition_physical_sites():
    doc = example()
    doc["slots"].extend(
        [
            slot("slot-03", "held"),
            slot("slot-04", "unknown", "unknown"),
            slot("slot-05", "stale", "unknown"),
            slot("slot-06", "confirmed", "occupied"),
        ]
    )
    data = current_status(snapshot(doc), NOW)
    assert data["availability_counts"] == {
        "physical_total": 5,
        "confirmed_free": 1,
        "confirmed_occupied": 1,
        "held": 1,
        "unknown": 1,
        "stale": 1,
    }
    assert not data["slots"][1]["confirmed_available"]


@pytest.mark.parametrize("view", ["mismatch", "unverifiable", "checking"])
def test_no_current_confirmation_with_unverified_view(view):
    doc = example()
    doc["camera_integrity"]["status"] = view
    data = current_status(snapshot(doc), NOW)
    assert data["freshness"] == "unverified"
    assert data["availability_counts"]["unknown"] == 1


@pytest.mark.parametrize(
    "health", ["interrupted", "eof", "stopped", "recovering", "waiting", "degraded"]
)
def test_offline_and_reconnect_cannot_resurrect_confirmation(health):
    doc = example()
    doc["source"]["health"] = health
    data = current_status(snapshot(doc), NOW)
    assert data["availability_counts"]["confirmed_free"] == 0
    assert data["availability_counts"]["confirmed_occupied"] == 0


def test_demo_evaluates_on_runner_clock_and_marks_itself_as_demo():
    doc = example("video")
    data = current_status(snapshot(doc), NOW + timedelta(days=14))
    assert data["mode"] == "demo"
    assert data["availability_counts"]["confirmed_free"] == 1
    assert data["publication_valid_until"] is None
    doc["source"]["health"] = "eof"
    assert (
        current_status(snapshot(doc), NOW)["availability_counts"]["confirmed_free"] == 0
    )


def test_history_bounds_order_session_changes_and_partial_tail(tmp_path):
    filename = tmp_path / "events.jsonl"
    data = example()

    def event(index, kind="observation"):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "event": kind,
                    "event_index": index,
                    "snapshot": data,
                    "observation": {"image": "/secret/location"},
                }
            )
            + "\n"
        )

    with filename.open("w") as log:
        for i in range(60):
            log.write(event(i))
        data["slots"][0] = slot(temporal="held")
        log.write(event(60, "heartbeat"))
        data["session_id"] = "test/run-1/1"
        log.write(event(61, "camera_integrity"))
        data["run_id"], data["session_id"] = "run-2", "test/run-2/0"
        log.write(event(0, "session_started"))
        log.write('{"schema_version":1, "event":')
    events = read_history(filename, 4)["events"]
    assert [e["event_index"] for e in events] == [59, 60, 61, 0]
    assert events[1]["slots"][0]["previous_temporal_status"] == "confirmed"
    assert events[1]["slots"][0]["temporal_status"] == "held"
    assert events[2]["session_reset"] and events[3]["session_reset"]
    assert all(row["slot_id"] == "slot-01" for e in events for row in e["slots"])
    assert "secret" not in json.dumps(events)


def test_reads_snapshots_while_the_runner_atomically_replaces_them(tmp_path):
    with LiveOutput(tmp_path) as writer:
        doc = example()
        writer.publish({"snapshot": doc}, {**doc, "sequence": 0})

        def publish():
            for i in range(18):
                data = {**doc, "sequence": i}
                writer.publish({"snapshot": data}, data)

        thread = Thread(target=publish)
        thread.start()
        try:
            for _ in range(60):
                observed = read_snapshot(tmp_path / "snapshot.json")
                assert observed["sequence"] in range(18)
        finally:
            thread.join()
    assert read_snapshot(tmp_path / "snapshot.json")["sequence"] == 17


def test_geometry_is_bound_to_exact_bytes_and_filters_diagnostic_slots(tmp_path):
    config = load_configuration(CONFIGURATION)
    from dataclasses import replace
    from hashlib import sha256

    projection = config.projections[0]
    irregular = Polygon(
        (
            Point(0.1, 0.1),
            Point(0.5, 0.15),
            Point(0.6, 0.3),
            Point(0.35, 0.45),
            Point(0.12, 0.32),
        )
    )
    edited = replace(
        config,
        projections=(replace(projection, polygon=irregular), *config.projections[1:]),
    )
    path = tmp_path / "geometry.json"
    save_configuration(edited, path)
    geometry = Geometry(path)
    data = snapshot(example())
    assert geometry.for_snapshot(data) is None
    data["configuration"]["sha256"] = sha256(path.read_bytes()).hexdigest()
    result = geometry.for_snapshot(data)
    assert result["image_size"] == [1280, 720]
    assert [(p["slot_id"], len(p["vertices"])) for p in result["projections"]] == [
        ("slot-01", 5)
    ]
    data["camera_id"] = "other"
    assert geometry.for_snapshot(data) is None


def test_http_status_history_health_and_read_only_surface(tmp_path):
    try:
        server = DashboardServer(0, tmp_path, clock=lambda: NOW)
    except PermissionError:
        pytest.skip("Local sockets are blocked in this execution context")
    with server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, method="GET", **headers):
            req = Request(base + path, method=method, headers=headers)
            try:
                return urlopen(req, timeout=2)
            except HTTPError as error:
                return error

        try:
            with request("/api/v1/health") as response:
                assert json.load(response)["status"] == "ok"
                assert response.headers["Cache-Control"] == "no-store"
            with request("/api/v1/status") as response:
                assert response.status == 404
            with request("/api/v1/history?limit=51") as response:
                assert response.status == 400
            filename = tmp_path / "snapshot.json"
            filename.write_text(json.dumps(example()))
            with request("/api/v1/status") as response:
                assert json.load(response)["availability_counts"]["confirmed_free"] == 1
            filename.write_text("corrupt")
            with request("/api/v1/status") as response:
                assert json.load(response)["error"] == "invalid_json"
                assert response.status == 422
            with request("/api/v1/status", "POST") as response:
                assert response.status == 405
            with request("/api/v1/restart") as response:
                assert response.status == 404
            with request("/api/v1/status", Host="not-loopback.example") as response:
                assert response.status == 403
            with request("/snapshot.json") as response:
                assert response.status == 404
        finally:
            server.shutdown()
            thread.join()

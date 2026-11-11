import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import observe
import replay

REPO = Path(__file__).resolve().parents[2]
BASE = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


class SyntheticModels:
    provenance = {"synthetic_test_fixture": True}

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, projections):
        return [0.95] * len(projections), []


@pytest.fixture
def frames(tmp_path):
    configuration = REPO / "apps/engine/examples/parking-setup-v1.json"
    image = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(image), np.zeros((720, 1280, 3), np.uint8))
    files = []
    for sequence in range(4):
        time = BASE + timedelta(seconds=sequence)
        args = argparse.Namespace(
            configuration=configuration,
            image=image,
            camera_id="camera-east",
            session_id="synthetic-session",
            sequence=sequence,
            captured_at=time,
            evaluated_at=time,
            max_age_seconds=60,
            view_confirmed=True,
            unreadable_slot_id=[],
            assets=tmp_path / "unused.json",
            device="cpu",
        )
        data = observe.observe(args, SyntheticModels)
        path = tmp_path / f"observation-{sequence}.json"
        observe.save_result(path, data)
        files.append(path)
    return files


def manifest(tmp_path, files, extra=(), **fields):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [{"observation": p.name} for p in files] + list(extra),
                **fields,
            }
        )
    )
    return path


def rewrite(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_actual_offline_contract_replays_deterministically_without_changing_sources(
    frames, tmp_path
):
    path = manifest(tmp_path, frames)
    before = {p: p.read_bytes() for p in frames}
    result = replay.replay(path, "slot-01")
    assert result == replay.replay(path, "slot-01")
    states = [row["temporal"] for row in result["snapshots"]]
    assert [state["estimated_state"] for state in states] == [
        "unknown",
        "unknown",
        "occupied",
        "occupied",
    ]
    final = states[-1]
    assert final["key"]["slot_id"] == "slot-01"
    assert final["key"]["configuration_revision"] == 1
    assert final["key"]["session_id"] == "synthetic-session"
    assert (
        final["last_observation"]["source_id"]
        == result["snapshots"][-1]["source_sha256"]
    )
    assert {p: p.read_bytes() for p in frames} == before


@pytest.mark.parametrize(
    "reason", ["model_conflict", "occluded", "leaf_shadow", "edge_of_frame"]
)
def test_unknown_reason_passes_through_without_reclassifying_scores(
    frames, tmp_path, reason
):
    def change(data):
        row = data["observations"][0]
        row.update(state="unknown", unknown_reason=reason, p_occupied=0.01)

    rewrite(frames[-1], change)
    path = manifest(tmp_path, frames, [{"tick_at": "2026-11-05T12:00:06+00:00"}])
    result = replay.replay(path, "slot-01")
    held = result["snapshots"][-2]
    assert held["raw_observation"]["unknown_reason"] == reason
    assert held["temporal"]["status"] == "held"
    assert result["snapshots"][-1]["temporal"]["estimated_state"] == "unknown"


def test_missing_slot_is_unknown_after_initial_known_identity(frames, tmp_path):
    rewrite(frames[-1], lambda data: data["observations"].pop(0))
    result = replay.replay(manifest(tmp_path, frames), "slot-01")
    assert result["snapshots"][-1]["raw_observation"]["unknown_reason"] == (
        "missing_slot_observation"
    )
    with pytest.raises(ValueError, match="selected slot_id"):
        replay.replay(manifest(tmp_path, frames), "invented-slot")


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_id", "camera-west"),
        ("parking_id", "other"),
        ("configuration_revision", 2),
        ("configuration_sha256", "c" * 64),
        ("source_session", "new-session"),
    ],
)
def test_mixed_camera_revision_session_is_an_error(frames, tmp_path, field, value):
    rewrite(frames[-1], lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match="Mixed stream"):
        replay.replay(manifest(tmp_path, frames), "slot-01")


def test_explicit_session_reset_and_motion_gate(frames, tmp_path):
    rewrite(frames[-1], lambda data: data.update(source_session="new", sequence=0))
    path = manifest(tmp_path, frames[:-1])
    data = json.loads(path.read_text())
    data["events"].append({"observation": frames[-1].name, "reset": True})
    path.write_text(json.dumps(data))
    result = replay.replay(path, "slot-01")
    state = result["snapshots"][-1]["temporal"]
    assert state["estimated_state"] == "unknown" and state["candidate"]["count"] == 1
    assert state["supporting_evidence"] is None
    data["events"][-1]["motion"] = "moving"
    path.write_text(json.dumps(data))
    state = replay.replay(path, "slot-01")["snapshots"][-1]["temporal"]
    assert state["reason"] == "motion_detected" and state["last_evidence"] is None


def test_duplicate_and_late_arrival_are_recorded_without_adding_confirmations(
    frames, tmp_path
):
    path = manifest(
        tmp_path,
        frames[:2],
        [
            {"observation": frames[1].name},
            {"observation": frames[0].name, "at": "2026-11-05T12:00:02+00:00"},
        ],
    )
    rows = replay.replay(path, "slot-01")["snapshots"]
    assert [row["disposition"] for row in rows] == [
        "accepted",
        "accepted",
        "duplicate",
        "out_of_order",
    ]
    assert rows[-1]["temporal"]["candidate"]["count"] == 2
    assert rows[-1]["temporal"]["estimated_state"] == "unknown"


def test_tick_disconnect_and_policy_are_explicit_in_output(frames, tmp_path):
    path = manifest(
        tmp_path,
        frames[:3],
        [
            {"tick_at": "2026-11-05T12:00:03+00:00"},
            {"disconnect_at": "2026-11-05T12:00:04+00:00"},
        ],
        policy={"unknown_hold_seconds": 1.5},
    )
    result = replay.replay(path, "slot-01")
    assert result["policy"]["unknown_hold_seconds"] == 1.5
    assert result["snapshots"][-2]["temporal"]["status"] == "held"
    assert result["snapshots"][-1]["temporal"]["reason"] == "camera_disconnected"


@pytest.mark.parametrize(
    "case", ["duplicate_slot", "schema", "state", "visibility", "view_bool"]
)
def test_malformed_observations_fail(frames, tmp_path, case):
    def change(data):
        if case == "duplicate_slot":
            data["observations"].append(copy.deepcopy(data["observations"][0]))
        elif case == "schema":
            data["schema_version"] = True
        elif case == "state":
            data["observations"][0]["state"] = "empty"
        elif case == "visibility":
            data["observations"][0]["visibility"] = "invisible"
        elif case == "view_bool":
            data["view_confirmed"] = "yes"

    rewrite(frames[0], change)
    with pytest.raises(ValueError):
        replay.replay(manifest(tmp_path, frames), "slot-01")


@pytest.mark.parametrize(
    "raw", ['{"schema_version":1,"schema_version":1}', '{"bad":NaN}']
)
def test_nonstandard_json_is_rejected(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        replay.read_json(path)


def test_cli_from_other_directory_is_byte_deterministic_and_exclusive(frames, tmp_path):
    path = manifest(tmp_path, frames)

    def run(output):
        return subprocess.run(
            [
                sys.executable,
                str(Path(replay.__file__).resolve()),
                "--manifest",
                str(path),
                "--slot-id",
                "slot-01",
                "--output",
                str(output),
            ],
            cwd="/tmp",
            text=True,
            capture_output=True,
        )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert run(first).returncode == 0
    assert run(second).returncode == 0
    assert first.read_bytes() == second.read_bytes()
    assert run(first).returncode == 2
    assert first.read_bytes() == second.read_bytes()


def test_cli_mixed_stream_has_no_partial_output(frames, tmp_path):
    rewrite(frames[-1], lambda data: data.update(camera_id="different"))
    path = manifest(tmp_path, frames)
    output = tmp_path / "failed.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(replay.__file__).resolve()),
            "--manifest",
            str(path),
            "--slot-id",
            "slot-01",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 2 and "Mixed stream" in process.stderr
    assert not output.exists()

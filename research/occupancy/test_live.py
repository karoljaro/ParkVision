from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from research.occupancy import live, replay
from research.occupancy.live_output import LiveOutput
from research.occupancy.models import CONVNEXT_SHA256, SLOT_SHA256
from research.occupancy.observe import ObservationSetup

from parkvision.capture.frame import Frame
from parkvision.capture.occupancy_source import FrameEvent

CONFIGURATION = (
    Path(__file__).resolve().parents[2] / "apps/engine/examples/parking-setup-v1.json"
)
ANCHOR = datetime(2020, 1, 1, tzinfo=timezone.utc)


def at(seconds):
    return ANCHOR + timedelta(seconds=seconds)


def new_session(*, diagnostic=("slot-02",)):
    return live.LiveSession(
        ObservationSetup.read(CONFIGURATION, "camera-east"),
        "synthetic-parking",
        "test",
        "acpds-rcnn",
        diagnostic,
    )


class Clock:
    now = ANCHOR

    def __call__(self):
        return self.now


class Events:
    time_source = "synthetic_test_clock"

    def __init__(self, events, clock=None):
        self.events = events
        self.clock = clock
        self.closed = False

    def __iter__(self):
        for event_at, event in self.events:
            if self.clock is not None:
                self.clock.now = event_at
            yield event

    def close(self):
        self.closed = True


class Model:
    provenance = {"per_slot": {"sha256": SLOT_SHA256}, "implementation": "test_mock"}

    def predict(self, frame, projections):
        score = 0.97 if frame.image[0, 0, 0] else 0.03
        return [score] * len(projections), []


def frame(seconds, occupied=False, *, seq=None, shape=(720, 1280, 3), skipped=0):
    return FrameEvent(
        "frame",
        Frame(
            "camera-east",
            int(seconds * 3) if seq is None else seq,
            at(seconds),
            np.full(shape, occupied, np.uint8),
        ),
        pts_seconds=seconds,
        skipped_frames=skipped,
        decode_seconds=0.0001,
    )


def execute(
    tmp_path, events, *, realtime=False, session=None, source=None, loader=Model
):
    session = session or new_session()
    clock = Clock()
    source = source or Events(events, clock if realtime else None)
    directory = tmp_path / "output"
    with LiveOutput(directory) as writer:
        stats = live.run_live(
            session,
            source,
            writer,
            loader,
            source_id="test",
            video_anchor=None if realtime else ANCHOR,
            view_confirmed=True,
            clock=source.clock or clock,
        )
    assert source.closed
    lines = (directory / "events.jsonl").read_text().splitlines()
    return stats, [json.loads(line) for line in lines], session


def observation_events(events):
    return [event for event in events if event["event"] == "observation"]


def counts(event):
    result = event["snapshot"]["availability_counts"]
    assert (
        sum(v for k, v in result.items() if k != "physical_total")
        == result["physical_total"]
    )
    return result


def test_occupancy_transitions_and_replay_match(tmp_path):
    events = [
        (at(i), frame(i, occupied=(i < 3 or i >= 6), skipped=2 * bool(i)))
        for i in range(9)
    ] + [(at(9), FrameEvent("eof"))]
    stats, logged, _ = execute(tmp_path, events)
    observations = observation_events(logged)
    assert counts(logged[0])["unknown"] == 1
    assert counts(observations[2])["confirmed_occupied"] == 1
    assert counts(observations[5])["confirmed_free"] == 1
    assert counts(observations[6])["held"] == 1
    assert counts(observations[6])["confirmed_free"] == 0
    assert observations[6]["snapshot"]["slots"][0]["estimated_state"] == "free"
    assert observations[5]["snapshot"]["slots"][1]["diagnostic_only"]
    assert not observations[5]["snapshot"]["slots"][1]["confirmed_available"]
    assert counts(observations[8])["confirmed_occupied"] == 1
    assert stats["skipped_frames"] == 16 and stats["processed_frames"] == 9
    assert logged[-1]["snapshot"]["source"]["health"] == "eof"
    assert counts(logged[-1])["unknown"] == 1

    manifest_events = []
    for i, event in enumerate(observations):
        path = tmp_path / f"observation-{i}.json"
        path.write_text(json.dumps(event["observation"]))
        manifest_events.append({"observation": path.name})
    manifest_events.append({"disconnect_at": logged[-1]["snapshot"]["updated_at"]})
    manifest_path = tmp_path / "replay.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "events": manifest_events})
    )
    outcome = replay.replay(manifest_path, "slot-01")
    for result, actual in zip(
        outcome["snapshots"], [*observations, logged[-1]], strict=True
    ):
        out = actual["snapshot"]["slots"][0]
        expected = result["temporal"]
        assert (
            out["estimated_state"],
            out["temporal_status"],
            out["temporal_reason"],
        ) == (expected["estimated_state"], expected["status"], expected["reason"])


def test_silence_expiry_and_session_reset_on_reconnect(tmp_path):
    events = [(at(i), frame(i)) for i in range(3)] + [
        (at(2.5), FrameEvent("heartbeat")),
        (at(4), FrameEvent("heartbeat")),
        (at(5), FrameEvent("heartbeat")),
        (at(7), FrameEvent("heartbeat")),
        (at(7.5), FrameEvent("disconnected", reason="signal_lost")),
        *[(at(i), frame(i, seq=i - 8)) for i in range(8, 11)],
    ]
    _, logged, _ = execute(tmp_path, events, realtime=True)
    hb = [e for e in logged if e["event"] == "heartbeat"]
    assert [e["snapshot"]["slots"][0]["temporal_status"] for e in hb] == [
        "held",
        "unknown",
        "stale",
    ]
    loss = next(e for e in logged if e["event"] == "source_lost")
    reset = next(e for e in logged if e["event"] == "source_recovered")
    assert counts(loss)["unknown"] == counts(reset)["unknown"] == 1
    assert loss["snapshot"]["session_id"] != reset["snapshot"]["session_id"]
    obs = observation_events(logged)
    assert [counts(e)["confirmed_free"] for e in obs] == [0, 0, 1, 0, 0, 1]


def test_resolution_mismatch_invalidates_current_view(tmp_path):
    events = [(at(i), frame(i)) for i in range(3)] + [
        (at(3), frame(3, shape=(360, 640, 3))),
        (at(4), frame(4)),
    ]
    _, logged, _ = execute(tmp_path, events)
    obs = observation_events(logged)
    assert counts(obs[2])["confirmed_free"] == 1
    assert obs[3]["observation"]["status"] == "dimension_mismatch"
    assert obs[4]["observation"]["status"] == "view_unverified"
    assert counts(obs[3])["unknown"] == counts(obs[4])["unknown"] == 1


def test_model_finishes_too_late_for_live_frame(tmp_path):
    clock = Clock()
    source = Events([(at(0), frame(0))], clock)

    class SlowModel(Model):
        def predict(self, frame, projections):
            clock.now = at(5)
            return super().predict(frame, projections)

    _, logged, _ = execute(tmp_path, [], source=source, realtime=True, loader=SlowModel)
    observation = observation_events(logged)[0]
    assert observation["observation"]["captured_at"] == at(0).isoformat()
    assert observation["observation"]["evaluated_at"] == at(5).isoformat()
    assert observation["observation"]["status"] == "stale_frame"
    assert counts(observation)["stale"] == 1


def test_previous_rtsp_generation_is_discarded_after_inference(tmp_path):
    class Connection(Events):
        current = True

        def is_current(self, event):
            return self.current

    source = Connection([(at(0), frame(0))], Clock())

    class DisconnectedModel(Model):
        def predict(self, frame, projections):
            source.current = False
            return super().predict(frame, projections)

    stats, logged, _ = execute(
        tmp_path, [], source=source, realtime=True, loader=DisconnectedModel
    )
    assert not observation_events(logged)
    assert stats["discarded_frames"] == 1
    lost = next(e for e in logged if e["event"] == "source_lost")
    assert (
        lost["snapshot"]["source"]["reason"]
        == "discarded_observation_from_interrupted_source"
    )
    assert counts(lost)["unknown"] == 1


def test_nonmonotonic_capture_fails_closed(tmp_path):
    events = [(at(i), frame(i)) for i in range(3)] + [(at(2), frame(2, seq=10))]
    stats, logged, _ = execute(tmp_path, events)
    rejection = next(e for e in logged if e["event"] == "frame_rejected")
    assert counts(rejection)["unknown"] == 1
    assert stats["discarded_frames"] == 1


@pytest.mark.parametrize(
    "mismatch", ["backend", "checkpoint", "configuration", "session"]
)
def test_changed_identity_does_not_mutate_temporal_state(tmp_path, mismatch):
    _, logged, _ = execute(tmp_path, [(at(0), frame(0))])
    doc = observation_events(logged)[0]["observation"]
    session = new_session()
    session.start(at(0))
    doc["source_session"] = session.session_id
    before = session.states.copy()
    if mismatch == "backend":
        doc["provenance"]["selected_model"] = "convnext-roi4"
    elif mismatch == "checkpoint":
        doc["provenance"]["models"]["per_slot"]["sha256"] = "a" * 64
    elif mismatch == "configuration":
        doc["configuration_sha256"] = "a" * 64
    else:
        doc["source_session"] = "previous-session"
    with pytest.raises(ValueError, match="Mixed|pinned"):
        session.apply(doc, at(0), 0)
    assert session.states == before


def test_requires_approved_geometry_and_expected_camera_and_parking(tmp_path):
    with pytest.raises(ValueError, match="Unknown camera"):
        ObservationSetup.read(CONFIGURATION, "missing-camera")
    setup = ObservationSetup.read(CONFIGURATION, "camera-east")
    with pytest.raises(ValueError, match="parking ID"):
        live.LiveSession(setup, "elsewhere", "test", "acpds-rcnn")
    draft = json.loads(CONFIGURATION.read_text())
    draft["approved_by"] = None
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="approved_by"):
        ObservationSetup.read(path, "camera-east")


def test_restart_overwrites_prior_confirmed_snapshot_before_loading_models(tmp_path):
    _, logged, _ = execute(tmp_path, [(at(i), frame(i)) for i in range(3)])
    latest_confirmed = observation_events(logged)[-1]
    folder = tmp_path / "output"
    # Model an abrupt old-process exit while its last publication was confirmed.
    with LiveOutput(folder) as writer:
        writer.publish(latest_confirmed, latest_confirmed["snapshot"])

    def start_model():
        snapshot = json.loads((folder / "snapshot.json").read_text())
        assert snapshot["availability_counts"]["confirmed_free"] == 0
        assert snapshot["availability_counts"]["unknown"] == 1
        assert snapshot["session_id"] != latest_confirmed["snapshot"]["session_id"]
        return Model()

    _, result, _ = execute(tmp_path, [(at(0), FrameEvent("eof"))], loader=start_model)
    assert len([item for item in result if item["event"] == "session_started"]) == 2


def test_interruption_during_model_load_publishes_stopped(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    stats, events, _ = execute(tmp_path, [], loader=interrupted)
    assert stats["model_load_seconds"] is None and stats["processed_frames"] == 0
    assert [event["event"] for event in events] == ["session_started", "stopped"]
    assert counts(events[-1])["unknown"] == 1


def test_video_decode_error_is_not_eof_and_clears_availability(tmp_path):
    events = [(at(i), frame(i)) for i in range(3)]
    events.append((at(3), FrameEvent("disconnected", reason="video_decode_failed")))
    with pytest.raises(RuntimeError, match="video_decode_failed"):
        execute(tmp_path, events)
    disk = json.loads((tmp_path / "output" / "snapshot.json").read_text())
    assert disk["source"]["health"] == "error"
    assert disk["source"]["reason"] == "video_decode_failed"
    assert disk["availability_counts"]["confirmed_free"] == 0


def test_external_view_mismatch_revokes_confirmation(tmp_path):
    session = new_session()
    source = Events([(at(i), frame(i)) for i in range(4)])
    directory = tmp_path / "output"
    with LiveOutput(directory) as writer:
        live.run_live(
            session,
            source,
            writer,
            Model,
            source_id="test",
            video_anchor=ANCHOR,
            view_confirmed=True,
            view_check=lambda candidate: candidate.sequence < 9,
        )
    events = [
        json.loads(x) for x in (directory / "events.jsonl").read_text().splitlines()
    ]
    before, after = observation_events(events)[-2:]
    assert counts(before)["confirmed_free"] == 1
    assert after["observation"]["status"] == "view_unverified"
    assert counts(after)["unknown"] == 1


def test_convnext_backend_is_pinned_through_the_live_path(tmp_path):
    class ConvNeXt(Model):
        provenance = {"per_slot": {"sha256": CONVNEXT_SHA256}}

    setup = ObservationSetup.read(CONFIGURATION, "camera-east")
    session = live.LiveSession(setup, "synthetic-parking", "conv", "convnext-roi4")
    _, events, _ = execute(
        tmp_path, [(at(0), frame(0))], session=session, loader=ConvNeXt
    )
    sample = observation_events(events)[0]
    assert sample["observation"]["provenance"]["selected_model"] == "convnext-roi4"
    assert sample["snapshot"]["model"]["checkpoint_sha256"] == CONVNEXT_SHA256

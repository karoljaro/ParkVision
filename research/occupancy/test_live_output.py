import json

import pytest

from research.occupancy import live_output
from research.occupancy.live_output import LiveOutput


def read_events(directory):
    path = directory / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_output_directory_has_single_writer_and_restart_appends_history(tmp_path):
    directory = tmp_path / "output"
    first = {"session_id": "before", "status": "confirmed"}
    with LiveOutput(directory) as writer:
        writer.publish({"event": "observation"}, first)
        with pytest.raises(BlockingIOError, match="Another live runner"):
            LiveOutput(directory)

    restarted = {"session_id": "after", "status": "unknown"}
    with LiveOutput(directory) as writer:
        writer.publish({"event": "startup"}, restarted)

    assert json.loads((directory / "snapshot.json").read_bytes()) == restarted
    assert read_events(directory) == [{"event": "observation"}, {"event": "startup"}]


@pytest.mark.parametrize(
    "tail, retained",
    [
        (b'{"status": "occ', []),
        (b'{"status":"occupied"}', [{"status": "occupied"}]),
        (b'{"status":"' + b"x" * 70000 + b"\xff", []),
    ],
)
def test_restart_repairs_only_unterminated_tail(tmp_path, tail, retained):
    completed = b'{"status":"unknown"}\n'
    path = tmp_path / "events.jsonl"
    path.write_bytes(completed + tail)

    with LiveOutput(tmp_path) as writer:
        writer.publish({"event": "startup"}, {"status": "unknown"})

    assert path.read_bytes().startswith(completed)
    assert read_events(tmp_path) == [
        {"status": "unknown"},
        *retained,
        {"event": "startup"},
    ]


def test_invalid_payload_leaves_journal_and_snapshot_unchanged(tmp_path):
    with LiveOutput(tmp_path) as writer:
        writer.publish({"event": "startup"}, {"status": "unknown"})
        output = [tmp_path / "events.jsonl", tmp_path / "snapshot.json"]
        before = [path.read_bytes() for path in output]
        with pytest.raises(ValueError, match="Out of range float"):
            writer.publish({"event": "observation"}, {"probability": float("nan")})
        with pytest.raises(TypeError, match="not JSON serializable"):
            writer.publish({"image": object()}, {"status": "unknown"})
        assert [path.read_bytes() for path in output] == before


def test_failed_atomic_replace_leaves_previous_snapshot_valid(tmp_path, monkeypatch):
    old = {"status": "unknown"}
    current = {"status": "confirmed", "parking": "Łąka"}
    with LiveOutput(tmp_path) as writer:
        writer.publish({"event": "startup"}, old)

        def interrupted_replace(source, target):
            assert source.parent == target.parent == tmp_path
            assert json.loads(source.read_bytes()) == current
            assert json.loads(target.read_bytes()) == old
            raise OSError("interrupted before replace")

        monkeypatch.setattr(live_output.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted before replace"):
            writer.publish({"event": "observation"}, current)

    assert json.loads((tmp_path / "snapshot.json").read_bytes()) == old
    assert read_events(tmp_path) == [{"event": "startup"}, {"event": "observation"}]
    assert not list(tmp_path.glob(".snapshot.*.tmp"))

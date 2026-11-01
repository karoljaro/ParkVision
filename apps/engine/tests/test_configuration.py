import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from parkvision.capture.frame import Frame
from parkvision.configuration import (
    ParkingConfiguration,
    load_configuration,
    save_configuration,
)
from parkvision.domain.camera import Camera
from parkvision.domain.parking_slot import ParkingSlot
from parkvision.domain.types import CameraId, ParkingId

ENGINE = Path(__file__).resolve().parents[1]
EXAMPLE = ENGINE / "examples" / "parking-setup-v1.json"


@pytest.fixture
def configuration() -> ParkingConfiguration:
    return load_configuration(EXAMPLE)


def test_round_trip_preserves_domain_objects_and_many_camera_mapping(
    tmp_path: Path, configuration: ParkingConfiguration
) -> None:
    original = EXAMPLE.read_bytes()
    path = tmp_path / "parking-r1.json"
    save_configuration(configuration, path)
    restored = load_configuration(path)

    assert restored == configuration
    assert EXAMPLE.read_bytes() == original
    assert len(restored.slots) == 2
    east = restored.projections_for_camera(
        CameraId("camera-east"), image_width=1280, image_height=720
    )
    west = restored.projections_for_camera(
        CameraId("camera-west"), image_width=1000, image_height=750
    )
    assert len(east) == 2
    assert len(west) == 1
    assert east[0].parking_slot_id == west[0].parking_slot_id
    assert east[0].polygon != west[0].polygon
    assert restored.camera_image_sizes == {
        "camera-east": (1280, 720),
        "camera-west": (1000, 750),
    }
    assert restored.approved_at == datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def test_frame_content_never_changes_saved_geometry(
    tmp_path: Path, configuration: ParkingConfiguration
) -> None:
    path = tmp_path / "parking-r1.json"
    save_configuration(configuration, path)
    saved_bytes = path.read_bytes()
    previous = None
    for sequence, brightness in enumerate((0, 127, 255)):
        frame = Frame(
            camera_id="camera-east",
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
            image=np.full((720, 1280, 3), brightness, dtype=np.uint8),
        )
        projections = configuration.projections_for_camera(
            CameraId(frame.camera_id),
            image_width=frame.image.shape[1],
            image_height=frame.image.shape[0],
        )
        if previous is not None:
            assert projections == previous
        previous = projections
    assert path.read_bytes() == saved_bytes
    assert load_configuration(path) == configuration


def test_export_never_overwrites_and_new_revision_keeps_ids(
    tmp_path: Path, configuration: ParkingConfiguration
) -> None:
    first = tmp_path / "parking-r1.json"
    second = tmp_path / "parking-r2.json"
    save_configuration(configuration, first)
    original = first.read_bytes()
    updated = replace(
        configuration,
        revision=2,
        approved_by="another-operator",
        approved_at=datetime(2026, 11, 2, 12, tzinfo=timezone.utc),
    )
    with pytest.raises(FileExistsError):
        save_configuration(updated, first)
    assert first.read_bytes() == original
    save_configuration(updated, second)
    restored = load_configuration(second)
    assert restored.revision == 2
    assert restored.approved_by == "another-operator"
    assert restored.projections == configuration.projections
    assert restored.slots == configuration.slots
    assert load_configuration(first).revision == 1


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("schema_version",), 2),
        (("schema_version",), True),
        (("revision",), 0),
        (("revision",), 1.5),
        (("revision",), True),
        (("approved_by",), ""),
        (("approved_at",), "2026-11-01T12:00:00"),
        (("approved_at",), "not-a-date"),
        (("parking", "id"), " parking "),
        (("cameras", 0, "id"), "camera-west"),
        (("cameras", 0, "parking_id"), "another-parking"),
        (("cameras", 0, "image_width"), 0),
        (("cameras", 0, "image_height"), 1),
        (("cameras", 0, "image_width"), 1280.0),
        (("cameras", 0, "image_height"), True),
        (("slots", 0, "id"), "slot-02"),
        (("slots", 0, "parking_id"), "another-parking"),
        (("slots", 0, "label"), 42),
        (("projections", 0, "camera_id"), "unknown-camera"),
        (("projections", 0, "parking_slot_id"), "unknown-slot"),
        (("projections", 0, "polygon", 0, "x"), -0.1),
        (("projections", 0, "polygon", 0, "y"), 1.1),
        (("projections", 0, "polygon", 0, "x"), True),
        (("projections", 0, "polygon", 0, "x"), "0.1"),
        (("projections", 0, "polygon", 0, "x"), float("nan")),
        (("projections", 0, "polygon", 0, "y"), float("inf")),
        (("projections", 0, "polygon", 0, "x"), 10**400),
        (("cameras",), []),
        (("slots",), []),
        (("projections",), []),
        (("projections",), {}),
    ],
)
def test_import_rejects_invalid_configuration(
    tmp_path: Path, location: tuple[str | int, ...], value: object
) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    target = data
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        load_configuration(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("remove_projection", [False, True])
def test_import_rejects_duplicate_pair_or_slot_without_projection(
    tmp_path: Path, remove_projection: bool
) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if remove_projection:
        data["projections"].pop()
    else:
        data["projections"].append(data["projections"][0])
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


@pytest.mark.parametrize(
    "vertices",
    [
        [(0.1, 0.1), (0.9, 0.9)],
        [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
        [(0.1, 0.1), (0.9, 0.1), (0.5, 0.8), (0.1, 0.1)],
        [(0.1, 0.1), (0.9, 0.9), (0.1, 0.8), (0.8, 0.1)],
    ],
)
def test_import_reuses_polygon_validation(
    tmp_path: Path, vertices: list[tuple[float, float]]
) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["projections"][0]["polygon"] = [{"x": x, "y": y} for x, y in vertices]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


@pytest.mark.parametrize("location", [(), ("cameras", 0), ("projections", 0)])
@pytest.mark.parametrize("missing", [False, True])
def test_import_rejects_unknown_and_missing_fields(
    tmp_path: Path, location: tuple[str | int, ...], missing: bool
) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    target = data
    for key in location:
        target = target[key]
    if missing:
        target.pop(next(iter(target)))
    else:
        target["unrecognized"] = "must not silently disappear"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


@pytest.mark.parametrize("payload", ['{"revision":1,"revision":2}', "[]", "{"])
def test_import_rejects_ambiguous_or_malformed_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


@pytest.mark.parametrize(
    ("camera_id", "width", "height"),
    [
        ("unknown", 1280, 720),
        ("camera-east", 720, 1280),
        ("camera-east", 640, 360),
        ("camera-east", True, 720),
    ],
)
def test_runtime_rejects_unknown_camera_or_different_dimensions(
    configuration: ParkingConfiguration, camera_id: str, width: int, height: int
) -> None:
    with pytest.raises(ValueError):
        configuration.projections_for_camera(
            CameraId(camera_id), image_width=width, image_height=height
        )


def test_direct_construction_checks_relations_and_snapshots_sizes(
    configuration: ParkingConfiguration,
) -> None:
    sizes = dict(configuration.camera_image_sizes)
    snapshot = replace(configuration, camera_image_sizes=sizes)
    sizes[CameraId("camera-east")] = (640, 360)
    assert snapshot.camera_image_sizes[CameraId("camera-east")] == (1280, 720)
    frozen_sizes = snapshot.camera_image_sizes
    with pytest.raises(TypeError):
        frozen_sizes[CameraId("camera-east")] = (640, 360)  # type: ignore[index]
    for sizes in ({}, {CameraId("camera-east"): (1280, 720)}):
        with pytest.raises(ValueError):
            replace(configuration, camera_image_sizes=sizes)
    with pytest.raises(ValueError, match="another parking"):
        replace(
            configuration,
            cameras=(Camera(CameraId("camera-east"), ParkingId("other"), "East"),),
            camera_image_sizes={CameraId("camera-east"): (1280, 720)},
        )
    with pytest.raises(ValueError, match="another parking"):
        replace(
            configuration,
            slots=(ParkingSlot(configuration.slots[0].id, ParkingId("other")),),
        )


@pytest.mark.parametrize("valid", [True, False])
def test_validation_cli_from_another_directory_is_read_only(
    tmp_path: Path, valid: bool
) -> None:
    path = tmp_path / "setup.json"
    path.write_bytes(EXAMPLE.read_bytes() if valid else b"{}")
    original = path.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parkvision.configuration",
            str(path),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ENGINE / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if valid else 2)
    assert (
        ("3 projections" in result.stdout)
        if valid
        else ("Invalid configuration" in result.stderr)
    )
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}

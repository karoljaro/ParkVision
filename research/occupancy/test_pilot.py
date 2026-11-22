from contextlib import contextmanager
from hashlib import sha256
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from research.occupancy import live, pilot
from research.occupancy.models import SLOT_SHA256
from research.occupancy.test_live import CONFIGURATION, Events, Model, at, frame


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def png(directory, filename, pixels):
    path = directory / filename
    assert cv2.imwrite(str(path), pixels)
    return {"path": filename, "sha256": digest(path)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    width, height = 800, 560
    data = json.loads(CONFIGURATION.read_bytes())
    data["cameras"][0].update(image_width=width, image_height=height)
    for row in data["projections"]:
        if row["camera_id"] == "camera-east":
            for point in row["polygon"]:
                point["x"] = 0.15 + point["x"] * 0.9
    configuration = tmp_path / "approved.json"
    configuration.write_text(json.dumps(data))
    y, x = np.indices((height, width))
    check = (((x // 90 + y // 90) % 2) * 255).astype(np.uint8)
    pixels = np.random.default_rng(31).integers(
        18, 230, (height, width), dtype=np.uint8
    )
    image = cv2.cvtColor(cv2.GaussianBlur(pixels, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)
    manifest = {
        "schema_version": 1,
        "camera_id": "camera-east",
        "configuration_sha256": digest(configuration),
        "configuration_revision": data["revision"],
        "image_size": [width, height],
        "source": "synthetic fixture",
        "approved_by": "test fixture",
        "approved_at": "2026-11-21T09:00:00+01:00",
        "reference": png(tmp_path, "ref.png", image),
        "fit_mask": png(tmp_path, "fit.png", cv2.bitwise_not(check)),
        "check_mask": png(tmp_path, "check.png", check),
    }
    view = tmp_path / "view.json"
    view.write_text(json.dumps(manifest))
    args = live.build_parser().parse_args(
        [
            "--rtsp-env",
            "--configuration",
            str(configuration),
            "--configuration-sha256",
            digest(configuration),
            "--configuration-revision",
            str(data["revision"]),
            "--view-reference",
            str(view),
            "--view-reference-sha256",
            digest(view),
            "--view-confirmed",
            "--parking-id",
            "synthetic-parking",
            "--camera-id",
            "camera-east",
            "--diagnostic-slot-id",
            "slot-02",
            "--session-id",
            "fixture",
            "--assets",
            str(tmp_path / "assets.json"),
            "--device",
            "cuda",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    args.authorized = True
    args.dashboard_port = 0
    args.alignment_preview = None
    source = SimpleNamespace(closed=False, is_current=lambda _: True)
    event = frame(0, shape=image.shape)
    event.frame.image[:] = image
    source.event = event

    @contextmanager
    def sample(_):
        try:
            yield source, source.event
        finally:
            source.closed = True

    def fake_models(*_, **__):
        return SimpleNamespace(
            provenance={
                "per_slot": {"sha256": SLOT_SHA256},
                "device_selection": {"selected": "cuda:0", "reason": "fixture"},
            },
        )

    monkeypatch.setattr(pilot, "sample_frame", sample)
    monkeypatch.setattr(pilot, "load_models", fake_models)
    return args, source


def test_ready_checks_real_output_and_dashboard(setup, tmp_path):
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
    except PermissionError as error:
        pytest.skip(str(error))
    args, source = setup
    args.alignment_preview = tmp_path / "alignment.png"
    result = pilot.preflight(args)
    assert result["status"] == "ready", result
    assert result["checks"]["configuration"]["physical_roi_count"] == 1
    assert result["checks"]["view"]["status"] == "verified"
    assert result["checks"]["dashboard"] == "loopback_http_verified"
    assert list(args.output_dir.iterdir()) == []
    assert args.alignment_preview.is_file()
    assert args.alignment_preview.stat().st_mode & 0o777 == 0o600
    assert source.closed


def test_unauthorized_does_not_connect(setup, monkeypatch):
    args, source = setup
    args.authorized = False
    monkeypatch.setattr(pilot, "sample_frame", lambda _: pytest.fail("opened source"))
    report = pilot.preflight(args)
    assert report["blockers"][0]["reason"] == "permission_not_confirmed"
    assert pilot.capture(args)["reason"] == "permission_not_confirmed"
    assert not source.closed
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"diagnostic_slot_id": ["slot-01", "slot-02"]}, "no_physical_roi"),
        ({"configuration_sha256": "f" * 64}, "configuration_pin_mismatch"),
        ({"view_reference_sha256": "f" * 64}, "reference_manifest_sha_mismatch"),
        ({"camera_id": "other-camera"}, "camera_id_mismatch"),
        ({"parking_id": "other-parking"}, "parking_id_mismatch"),
    ],
)
def test_rejects_identity_or_geometry(setup, change, reason):
    args, source = setup
    vars(args).update(change)
    result = pilot.preflight(args)
    assert result["status"] == "blocked", result
    assert result["reason"] == reason
    assert not source.closed


@pytest.mark.parametrize(
    "edit, reason",
    [
        ("draft", "geometry_approval_required"),
        ("reference_missing", "reference_file_missing"),
        ("mask_missing", "reference_file_missing"),
        ("mask_corrupt", "invalid_reference_or_mask"),
        ("mask_hash", "reference_asset_sha_mismatch"),
        ("weak_support", "insufficient_roi_spatial_support"),
    ],
)
def test_rejects_bad_approval_or_reference(setup, tmp_path, edit, reason):
    args, source = setup
    doc = json.loads(args.view_reference.read_bytes())
    check = tmp_path / doc["check_mask"]["path"]
    if edit == "draft":
        config = json.loads(args.configuration.read_bytes())
        config.update(approved_by=None, approved_at=None)
        args.configuration.write_text(json.dumps(config))
        args.configuration_sha256 = digest(args.configuration)
    elif edit == "reference_missing":
        args.view_reference.unlink()
    elif edit == "mask_missing":
        check.unlink()
    elif edit in {"mask_corrupt", "mask_hash"}:
        check.write_bytes(b"not a PNG")
        if edit == "mask_corrupt":
            doc["check_mask"]["sha256"] = digest(check)
    else:
        width, height = doc["image_size"]
        fit = np.zeros((height, width), np.uint8)
        check_mask = np.zeros_like(fit)
        fit[:, :550] = 255
        check_mask[:, 600:] = 255
        doc["fit_mask"] = png(tmp_path, "fit-weak.png", fit)
        doc["check_mask"] = png(tmp_path, "check-weak.png", check_mask)
    if args.view_reference.exists():
        args.view_reference.write_text(json.dumps(doc))
        args.view_reference_sha256 = digest(args.view_reference)
    result = pilot.preflight(args)
    assert result["status"] == "blocked", result
    assert result["reason"] == reason
    assert not source.closed


@pytest.mark.parametrize(
    "change, status, reason",
    [
        ("resize", "mismatch", "resolution_mismatch"),
        ("shift", "mismatch", "camera_view_changed"),
        ("flat", "unverifiable", "insufficient_"),
    ],
)
def test_checks_current_view(setup, change, status, reason):
    args, source = setup
    image = source.event.frame.image
    height, width = image.shape[:2]
    if change == "resize":
        source.event = frame(0, shape=image[1:].shape)
    elif change == "shift":
        image[:] = cv2.warpAffine(
            image, np.float32([[1, 0, 5], [0, 1, 0]]), (width, height)
        )
    else:
        image.fill(80)
    result = pilot.preflight(args)
    assert result["status"] == "blocked", result
    assert result["integrity_status"] == status
    assert result["reason"].startswith(reason)
    assert source.closed


def test_forced_cuda_is_blocked(setup, monkeypatch):
    import torch

    args, _ = setup
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from research.occupancy.models import resolve_device

    def load(_, device, model):
        resolve_device(device)
        pytest.fail("CUDA should be blocked")

    monkeypatch.setattr(pilot, "load_models", load)
    result = pilot.preflight(args)
    assert result["reason"] == "cuda_unavailable"


def test_existing_output_is_preserved(setup):
    args, _ = setup
    args.output_dir.mkdir()
    snapshot = args.output_dir / "snapshot.json"
    snapshot.write_bytes(b"existing session")
    result = pilot.preflight(args)
    assert result["reason"] == "output_directory_not_empty"
    assert snapshot.read_bytes() == b"existing session"


def test_rtsp_failure_keeps_credentials_out_of_native_and_cli_output(tmp_path):
    # A refused connection must finish after the first attempt and omit the URL.
    secret = "private-fixture-password"
    path_token = "private-fixture-path"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.occupancy.pilot",
            "capture",
            "--rtsp-env",
            "--authorized",
            "--camera-id",
            "camera-east",
            "--image",
            str(tmp_path / "candidate.png"),
        ],
        env={
            **os.environ,
            "PARKVISION_RTSP": f"rtsp://operator:{secret}@127.0.0.1:1/{path_token}",
        },
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["reason"] == "rtsp_open_failed"
    assert all(
        part not in result.stdout + result.stderr for part in (secret, path_token)
    )
    assert not (tmp_path / "candidate.png").exists()


def test_rtsp_session_uses_logical_source_id_in_json(tmp_path, monkeypatch):
    from parkvision.capture import occupancy_source

    fake = Events([])
    fake.time_source = occupancy_source.RTSP_CLOCK
    source_url = "rtsp://operator:private-fixture@host.example/device?token=fixture"
    monkeypatch.setenv("PARKVISION_RTSP", source_url)
    monkeypatch.setattr(occupancy_source, "RTSPSource", lambda url, *a, **kw: fake)
    args = live.build_parser().parse_args(
        [
            "--rtsp-env",
            "--configuration",
            str(CONFIGURATION),
            "--parking-id",
            "synthetic-parking",
            "--camera-id",
            "camera-east",
            "--session-id",
            "fixture",
            "--view-confirmed",
            "--assets",
            str(tmp_path / "assets.json"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    source, source_id = live.build_source(args)
    assert "PARKVISION_RTSP" not in os.environ
    with pilot.LiveOutput(tmp_path) as writer:
        live.run_live(
            live.prepare_session(args),
            source,
            writer,
            Model,
            source_id=source_id,
            view_confirmed=True,
            clock=lambda: at(0),
        )
    for filename in ("snapshot.json", "events.jsonl"):
        content = (tmp_path / filename).read_text()
        assert "rtsp/camera-east" in content
        assert all(
            part not in content
            for part in ("private-fixture", "host.example", "?token=")
        )

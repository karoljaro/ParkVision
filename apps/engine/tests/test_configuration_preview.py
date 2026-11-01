import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from parkvision.configuration import ParkingConfiguration, load_configuration
from parkvision.configuration_preview import (
    render_configuration_preview,
    save_configuration_preview,
)
from parkvision.domain.camera import Camera
from parkvision.domain.geometry.point import Point
from parkvision.domain.geometry.polygon import Polygon
from parkvision.domain.slot_projection import SlotProjection
from parkvision.domain.types import CameraId

ENGINE = Path(__file__).resolve().parents[1]
EXAMPLE = ENGINE / "examples" / "parking-setup-v1.json"


@pytest.fixture
def configuration() -> ParkingConfiguration:
    return load_configuration(EXAMPLE)


@pytest.mark.parametrize(
    ("camera_id", "shape", "expected_count"),
    [("camera-east", (720, 1280, 3), 2), ("camera-west", (750, 1000, 3), 1)],
)
def test_camera_selection_labels_and_input_immutability(
    configuration: ParkingConfiguration,
    monkeypatch: pytest.MonkeyPatch,
    camera_id: str,
    shape: tuple[int, int, int],
    expected_count: int,
) -> None:
    image = np.full(shape, 100, dtype=np.uint8)
    image.flags.writeable = False
    texts: list[str] = []
    strokes: list[object] = []
    put_text, polylines = cv2.putText, cv2.polylines

    def draw_text(image, text, *args, **kwargs):
        texts.append(text)
        return put_text(image, text, *args, **kwargs)

    def draw_polygons(image, points, closed, color, *args, **kwargs):
        strokes.append(color)
        return polylines(image, points, closed, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", draw_text)
    monkeypatch.setattr(cv2, "polylines", draw_polygons)
    preview = render_configuration_preview(configuration, CameraId(camera_id), image)

    assert len(strokes) == 2 * expected_count  # border and colored contour per ROI
    assert '[1] id: "slot-01"' in texts
    assert 'label: "01"' in texts
    assert ('[2] id: "slot-02"' in texts) == (camera_id == "camera-east")
    assert ("label: (none)" in texts) == (camera_id == "camera-east")
    assert f'camera: "{camera_id}"' in texts
    assert preview.shape[0] >= shape[0]
    assert preview.shape[1] > shape[1]
    assert np.all(image == 100)
    assert not np.shares_memory(preview, image)
    assert np.any(preview[: shape[0], : shape[1]] != image)
    assert np.all(preview[shape[0] - 1, shape[1] - 1] == image[-1, -1])
    assert configuration == load_configuration(EXAMPLE)


def test_extreme_coordinates_use_width_minus_one_and_height_minus_one(
    configuration: ParkingConfiguration, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = configuration.slots[0]
    projection = SlotProjection(
        CameraId("camera-east"),
        slot.id,
        Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))),
    )
    configuration = replace(
        configuration,
        slots=(slot,),
        projections=(projection,),
        camera_image_sizes={
            CameraId("camera-east"): (101, 81),
            **{
                camera: size
                for camera, size in configuration.camera_image_sizes.items()
                if camera != "camera-east"
            },
        },
    )
    calls = []
    polylines = cv2.polylines

    def draw(image, points, *args, **kwargs):
        calls.append(points[0].copy())
        assert kwargs["shift"] == 8
        return polylines(image, points, *args, **kwargs)

    monkeypatch.setattr(cv2, "polylines", draw)
    preview = render_configuration_preview(
        configuration, CameraId("camera-east"), np.full((81, 101, 3), 80, np.uint8)
    )
    expected = np.array([(0, 0), (25600, 0), (25600, 20480), (0, 20480)])
    assert np.array_equal(calls[0].reshape(-1, 2), expected)
    assert np.any(preview[80, 100] != 80)


def test_shared_slot_keeps_same_id_badge_and_color_in_both_cameras(
    configuration: ParkingConfiguration, monkeypatch: pytest.MonkeyPatch
) -> None:
    colors = []
    numbers = []
    polylines, put_text = cv2.polylines, cv2.putText

    def draw(image, points, closed, color, thickness, *args, **kwargs):
        if thickness == 2:
            colors.append(color)
        return polylines(image, points, closed, color, thickness, *args, **kwargs)

    def text(image, value, *args, **kwargs):
        numbers.append(value)
        return put_text(image, value, *args, **kwargs)

    monkeypatch.setattr(cv2, "polylines", draw)
    monkeypatch.setattr(cv2, "putText", text)
    for camera_id in (CameraId("camera-east"), CameraId("camera-west")):
        width, height = configuration.camera_image_sizes[camera_id]
        render_configuration_preview(
            configuration, camera_id, np.zeros((height, width, 3), np.uint8)
        )
    assert colors[0] == colors[2]
    assert numbers.count("1") == 2
    assert numbers.count('[1] id: "slot-01"') == 2


def test_no_projection_does_not_invent_an_roi(
    configuration: ParkingConfiguration,
) -> None:
    camera_id = CameraId("unconfigured-view")
    configuration = replace(
        configuration,
        cameras=configuration.cameras
        + (Camera(camera_id, configuration.parking.id, "Bez ROI"),),
        camera_image_sizes={**configuration.camera_image_sizes, camera_id: (60, 40)},
    )
    image = np.full((40, 60, 3), 60, np.uint8)
    preview = render_configuration_preview(configuration, camera_id, image)
    assert np.array_equal(preview[:40, :60], image)


def test_unicode_long_labels_and_controls_stay_in_legend(
    configuration: ParkingConfiguration, monkeypatch: pytest.MonkeyPatch
) -> None:
    label = "Łódź / Zażółć gęślą jaźń\n" + "W" * 600
    configuration = replace(
        configuration,
        slots=(replace(configuration.slots[0], label=label), configuration.slots[1]),
    )
    texts = []
    put_text = cv2.putText

    def draw(image, text, *args, **kwargs):
        texts.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", draw)
    preview = render_configuration_preview(
        configuration, CameraId("camera-east"), np.zeros((720, 1280, 3), np.uint8)
    )
    assert any("Łódź / Zażółć gęślą jaźń\\n" in text for text in texts)
    assert preview.shape[0] > 720
    # Wrapped text leaves right/bottom margins instead of getting clipped.
    assert np.all(preview[:, -8:] == 28)
    assert np.all(preview[-8:] == 28)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((720, 1280), np.uint8),
        np.zeros((720, 1280, 4), np.uint8),
        np.zeros((720, 1280, 3), np.float32),
        np.zeros((0, 1280, 3), np.uint8),
        np.zeros((360, 640, 3), np.uint8),
    ],
)
def test_renderer_rejects_invalid_image_or_resolution(
    configuration: ParkingConfiguration, image: np.ndarray
) -> None:
    with pytest.raises((ValueError, TypeError)):
        render_configuration_preview(configuration, CameraId("camera-east"), image)


@pytest.fixture
def input_paths(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "setup.json"
    config_path.write_bytes(EXAMPLE.read_bytes())
    image_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(image_path), np.full((720, 1280, 3), 90, np.uint8))
    return config_path, image_path


def test_png_export_is_lossless_and_preserves_input_files(
    tmp_path: Path, input_paths: tuple[Path, Path]
) -> None:
    config_path, image_path = input_paths
    original = (config_path.read_bytes(), image_path.read_bytes())
    output = tmp_path / "preview.png"
    save_configuration_preview(config_path, image_path, CameraId("camera-east"), output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imread(str(output))
    assert decoded is not None
    assert decoded.shape[0] >= 720 and decoded.shape[1] > 1280
    assert np.all(decoded[719, 1279] == 90)
    assert original == (config_path.read_bytes(), image_path.read_bytes())


@pytest.mark.parametrize(
    "alias", ["existing", "image", "config", "symlink", "hardlink"]
)
def test_export_never_overwrites_existing_files_or_input_aliases(
    tmp_path: Path, input_paths: tuple[Path, Path], alias: str
) -> None:
    config_path, image_path = input_paths
    output = tmp_path / "preview.png"
    if alias == "image":
        output = image_path
    elif alias == "config":
        config_path = tmp_path / "setup-with-png-suffix.png"
        config_path.write_bytes(EXAMPLE.read_bytes())
        output = config_path
    elif alias == "symlink":
        output.symlink_to(image_path)
    elif alias == "hardlink":
        output.hardlink_to(image_path)
    else:
        output.write_bytes(b"existing preview")
    originals = (config_path.read_bytes(), image_path.read_bytes(), output.read_bytes())
    with pytest.raises(FileExistsError):
        save_configuration_preview(
            config_path, image_path, CameraId("camera-east"), output
        )
    assert originals == (
        config_path.read_bytes(),
        image_path.read_bytes(),
        output.read_bytes(),
    )


@pytest.mark.parametrize(
    "problem", ["camera", "size", "json", "missing", "decode", "suffix"]
)
def test_invalid_inputs_leave_no_output(
    tmp_path: Path, input_paths: tuple[Path, Path], problem: str
) -> None:
    config_path, image_path = input_paths
    output = tmp_path / ("preview.jpg" if problem == "suffix" else "preview.png")
    camera_id = CameraId("unknown" if problem == "camera" else "camera-east")
    if problem == "size":
        assert cv2.imwrite(str(image_path), np.zeros((720, 1279, 3), np.uint8))
    elif problem == "json":
        config_path.write_text("{}", encoding="utf-8")
    elif problem == "missing":
        image_path.unlink()
    elif problem == "decode":
        image_path.write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError)):
        save_configuration_preview(config_path, image_path, camera_id, output)
    assert not output.exists()


def test_encoder_failure_leaves_no_output(
    tmp_path: Path, input_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, image_path = input_paths
    output = tmp_path / "preview.png"
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, np.array([], np.uint8)))
    with pytest.raises(OSError, match="encode"):
        save_configuration_preview(
            config_path, image_path, CameraId("camera-east"), output
        )
    assert not output.exists()


@pytest.mark.parametrize("camera_id", ["camera-east", "unknown"])
def test_cli_from_another_directory(
    tmp_path: Path, input_paths: tuple[Path, Path], camera_id: str
) -> None:
    config_path, image_path = input_paths
    originals = config_path.read_bytes(), image_path.read_bytes()
    output = tmp_path / "preview.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parkvision.configuration_preview",
            "--configuration",
            str(config_path),
            "--camera-id",
            camera_id,
            "--image",
            str(image_path),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ENGINE / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if camera_id == "camera-east" else 2)
    assert output.exists() == (camera_id == "camera-east")
    assert "Traceback" not in result.stderr
    assert originals == (config_path.read_bytes(), image_path.read_bytes())

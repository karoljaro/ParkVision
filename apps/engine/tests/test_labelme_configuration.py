import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

from parkvision.configuration import load_configuration, save_configuration
from parkvision.configuration_preview import render_configuration_preview
from parkvision.domain.geometry.point import Point
from parkvision.labelme_configuration import (
    approve_labelme_configuration,
    load_labelme_configuration,
    main,
)

ENGINE = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(image_path), np.full((81, 101, 3), 100, np.uint8))
    data: dict[str, object] = {
        "version": "5.9.1",
        "flags": {},
        "shapes": [
            {
                "label": "stable-slot-alpha",
                "description": "Pilot Ł",
                "points": [[0, 0], [50, 0], [50, 80], [0, 80]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            },
            {
                "label": "stable-slot-beta",
                "description": "",
                "points": [[50, 0], [100, 0], [100, 80], [50, 80]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": "reference.png",
        "imageData": None,
        "imageWidth": 101,
        "imageHeight": 81,
        "parkvision": {
            "task": "parking-slot-setup",
            "parking_id": "synthetic-parking",
            "parking_name": "Synthetic test",
            "camera_id": "synthetic-camera",
            "camera_name": "Test view",
            "revision": 1,
            "image_sha256": sha256(image_path.read_bytes()).hexdigest(),
        },
    }
    annotation = tmp_path / "reference.json"
    annotation.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return annotation, image_path


def test_labelme_converts_to_existing_models_with_exact_coordinates_and_stable_ids(
    inputs: tuple[Path, Path],
) -> None:
    annotation, image_path = inputs
    before = annotation.read_bytes(), image_path.read_bytes()
    draft = load_labelme_configuration(annotation, image_path)
    assert not draft.is_approved
    assert draft.approved_by is None and draft.approved_at is None
    assert draft.parking.id == "synthetic-parking"
    assert draft.cameras[0].id == "synthetic-camera"
    assert draft.camera_image_sizes == {"synthetic-camera": (101, 81)}
    assert draft.slots[0].id == "stable-slot-alpha"
    assert draft.slots[0].label == "Pilot Ł"
    assert draft.slots[1].label is None
    assert draft.projections[0].polygon.vertices == (
        Point(0, 0),
        Point(0.5, 0),
        Point(0.5, 1),
        Point(0, 1),
    )
    assert draft.projections[1].polygon.area == 0.5
    assert before == (annotation.read_bytes(), image_path.read_bytes())

    data = json.loads(annotation.read_text(encoding="utf-8"))
    data["shapes"].reverse()
    annotation.write_text(json.dumps(data), encoding="utf-8")
    reordered = load_labelme_configuration(annotation, image_path)
    assert {p.parking_slot_id: p.polygon for p in draft.projections} == {
        p.parking_slot_id: p.polygon for p in reordered.projections
    }


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("imagePath",), "another-image.png"),
        (("imageData",), "embedded data"),
        (("imageWidth",), 100),
        (("imageHeight",), True),
        (("imageWidth",), 101.0),
        (("parkvision", "task"), "paint-segmentation"),
        (("parkvision", "camera_id"), ""),
        (("parkvision", "parking_id"), 5),
        (("parkvision", "revision"), 0),
        (("parkvision", "image_sha256"), "0" * 64),
        (("parkvision", "image_sha256"), "not-a-hash"),
        (("shapes",), []),
        (("shapes", 1, "label"), "stable-slot-alpha"),
        (("shapes", 0, "shape_type"), "rectangle"),
        (("shapes", 0, "mask"), "embedded mask"),
        (("shapes", 0, "group_id"), 1),
        (("shapes", 0, "description"), True),
        (("shapes", 0, "points", 0, 0), True),
        (("shapes", 0, "points", 0, 0), "0"),
        (("shapes", 0, "points", 0, 0), float("nan")),
        (("shapes", 0, "points", 0, 0), float("inf")),
        (("shapes", 0, "points", 0, 0), 10**400),
        (("shapes", 0, "points", 0, 0), -1),
        (("shapes", 0, "points", 0, 0), 101),
        (("shapes", 0, "points", 0), [0, 0, 0]),
        (("shapes", 0, "points"), [[0, 0], [50, 0]]),
        (("shapes", 0, "points"), [[0, 0], [50, 0], [50, 80], [0, 0]]),
        (("shapes", 0, "points"), [[0, 0], [50, 80], [0, 70], [40, 0]]),
    ],
)
def test_import_rejects_bad_labelme_data(
    inputs: tuple[Path, Path], location: tuple[str | int, ...], value: object
) -> None:
    annotation, image_path = inputs
    data = json.loads(annotation.read_text(encoding="utf-8"))
    target = data
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    annotation.write_text(json.dumps(data), encoding="utf-8")
    original = annotation.read_bytes(), image_path.read_bytes()
    with pytest.raises(ValueError):
        load_labelme_configuration(annotation, image_path)
    assert original == (annotation.read_bytes(), image_path.read_bytes())


@pytest.mark.parametrize("key", ["parking_id", "camera_id", "slot_id"])
def test_rejects_conflicting_shape_relationships(
    inputs: tuple[Path, Path], key: str
) -> None:
    annotation, image_path = inputs
    data = json.loads(annotation.read_text(encoding="utf-8"))
    data["shapes"][0][key] = "other-object"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="relationships"):
        load_labelme_configuration(annotation, image_path)


def test_paint_annotations_and_duplicate_json_keys_are_not_accepted(
    inputs: tuple[Path, Path],
) -> None:
    annotation, image_path = inputs
    data = json.loads(annotation.read_text(encoding="utf-8"))
    del data["parkvision"]
    data["shapes"][0]["label"] = "paint"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_labelme_configuration(annotation, image_path)
    annotation.write_text('{"shapes": [], "shapes": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_labelme_configuration(annotation, image_path)


def test_changed_image_is_rejected_even_at_same_resolution(
    inputs: tuple[Path, Path],
) -> None:
    annotation, image_path = inputs
    assert cv2.imwrite(str(image_path), np.zeros((81, 101, 3), np.uint8))
    with pytest.raises(ValueError, match="image_sha256"):
        load_labelme_configuration(annotation, image_path)


def test_draft_cannot_be_exported_or_loaded_as_approved(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    draft = load_labelme_configuration(*inputs)
    output = tmp_path / "parking-r1.json"
    with pytest.raises(ValueError, match="DRAFT"):
        save_configuration(draft, output)
    assert not output.exists()
    approved_example = ENGINE / "examples/parking-setup-v1.json"
    data = json.loads(approved_example.read_text(encoding="utf-8"))
    data["approved_by"] = data["approved_at"] = None
    output.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(output)
    with pytest.raises(ValueError, match="both"):
        replace(draft, approved_by="operator")
    with pytest.raises(ValueError, match="both"):
        replace(draft, approved_at=datetime.now(timezone.utc))


def test_explicit_approval_round_trip_and_stale_review_block(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    annotation, image_path = inputs
    original = annotation.read_bytes(), image_path.read_bytes()
    reviewed_hash = sha256(original[0]).hexdigest()
    output = tmp_path / "parking-r1.json"

    def approve_to(path: Path) -> None:
        approve_labelme_configuration(
            annotation,
            image_path,
            path,
            reviewed_annotation_sha256=reviewed_hash,
            approved_by="synthetic-reviewer",
            approved_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        )

    approve_to(output)
    approved = load_configuration(output)
    draft = load_labelme_configuration(annotation, image_path)
    assert approved.is_approved
    assert approved.projections == draft.projections
    assert approved.slots == draft.slots
    assert approved.approved_by == "synthetic-reviewer"
    with pytest.raises(FileExistsError):
        approve_to(output)
    assert original == (annotation.read_bytes(), image_path.read_bytes())
    annotation.write_bytes(original[0] + b"\n")
    stale_output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="reviewed SHA256"):
        approve_to(stale_output)
    assert not stale_output.exists()


def test_draft_preview_has_visible_warning_in_photo_and_legend(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = load_labelme_configuration(*inputs)
    texts = []
    put_text = cv2.putText

    def draw(image, text, *args, **kwargs):
        texts.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", draw)
    render_configuration_preview(
        draft, draft.cameras[0].id, np.zeros((81, 101, 3), np.uint8)
    )
    assert texts.count("DRAFT - NOT APPROVED") == 2


def test_cli_preview_binds_printed_hash_to_rendered_annotation(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    annotation, image_path = inputs
    digest = sha256(annotation.read_bytes()).hexdigest()

    def concurrent_edit(*args):
        annotation.write_bytes(annotation.read_bytes() + b"\n")

    monkeypatch.setattr(
        "parkvision.labelme_configuration.save_geometry_preview", concurrent_edit
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "labelme_configuration",
            "preview",
            "--annotation",
            str(annotation),
            "--image",
            str(image_path),
            "--output",
            str(tmp_path / "draft.png"),
        ],
    )
    main()
    assert f"Annotation SHA256: {digest}" in capsys.readouterr().out
    assert sha256(annotation.read_bytes()).hexdigest() != digest


@pytest.mark.parametrize("approve", [False, True])
def test_cli_preview_or_missing_approval_does_not_export_final_configuration(
    tmp_path: Path, inputs: tuple[Path, Path], approve: bool
) -> None:
    annotation, image_path = inputs
    output = tmp_path / ("parking-r1.json" if approve else "draft.png")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parkvision.labelme_configuration",
            "approve" if approve else "preview",
            "--annotation",
            str(annotation),
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
    assert result.returncode == (2 if approve else 0)
    assert output.exists() is not approve
    if not approve:
        assert "DRAFT - NOT APPROVED" in result.stdout
        assert sha256(annotation.read_bytes()).hexdigest() in result.stdout


def test_cli_explicit_approval_and_final_preview_on_synthetic_data(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    annotation, image_path = inputs
    original = annotation.read_bytes(), image_path.read_bytes()
    output = tmp_path / "approved.json"
    approved_preview = tmp_path / "approved-preview.png"
    environment = {**os.environ, "PYTHONPATH": str(ENGINE / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parkvision.labelme_configuration",
            "approve",
            "--annotation",
            str(annotation),
            "--image",
            str(image_path),
            "--output",
            str(output),
            "--reviewed-annotation-sha256",
            sha256(original[0]).hexdigest(),
            "--approved-by",
            "synthetic-reviewer",
            "--approved-at",
            "2026-11-01T12:00:00+00:00",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert load_configuration(output).is_approved
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parkvision.configuration_preview",
            "--configuration",
            str(output),
            "--camera-id",
            "synthetic-camera",
            "--image",
            str(image_path),
            "--output",
            str(approved_preview),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert approved_preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert original == (annotation.read_bytes(), image_path.read_bytes())

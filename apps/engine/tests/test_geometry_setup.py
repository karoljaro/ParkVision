import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from parkvision.configuration import load_configuration
from parkvision.domain.geometry.point import Point
from parkvision.domain.types import CameraId
from parkvision.labelme_configuration import load_labelme_configuration

_ENGINE = Path(__file__).resolve().parents[1]
_CAMERA = CameraId("geometry-camera")


def _run(
    cwd: Path, module: str, *arguments: str | Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", f"parkvision.{module}", *map(str, arguments)],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(_ENGINE / "src")},
        capture_output=True,
        text=True,
        check=False,
    )


def _propose(image: Path, output: Path, revision: int = 1) -> None:
    result = _run(
        image.parent,
        "geometry_setup",
        "propose",
        "--image",
        image,
        "--output-dir",
        output,
        "--parking-id",
        "geometry-test-lot",
        "--parking-name",
        "Geometry test lot",
        "--camera-id",
        _CAMERA,
        "--camera-name",
        "Geometry test camera",
        "--revision",
        str(revision),
    )
    assert result.returncode == 0, result.stderr


def _approve(
    image: Path,
    draft: Path,
    output: Path,
    digest: str,
) -> subprocess.CompletedProcess[str]:
    return _run(
        image.parent,
        "labelme_configuration",
        "approve",
        "--image",
        image,
        "--annotation",
        draft,
        "--output",
        output,
        "--reviewed-annotation-sha256",
        digest,
        "--approved-by",
        "synthetic-reviewer",
        "--approved-at",
        "2026-11-12T12:00:00+01:00",
    )


def test_cli_review_edit_reject_and_stable_ids(
    tmp_path: Path,
    geometry_row_image: NDArray[np.uint8],
) -> None:
    image = tmp_path / "geometry reference.png"
    assert cv2.imwrite(str(image), geometry_row_image)
    reference = image.read_bytes()
    output = tmp_path / "proposed"
    _propose(image, output, revision=2)

    draft_path = output / "draft.json"
    draft_bytes = draft_path.read_bytes()
    original_hash = sha256(draft_bytes).hexdigest()
    evidence = json.loads((output / "evidence.json").read_bytes())
    assert evidence["initial_draft_sha256"] == original_hash
    assert evidence["image_sha256"] == sha256(reference).hexdigest()
    assert cv2.imread(str(output / "preview.png")) is not None
    draft = json.loads(draft_bytes)
    labels = [shape["label"] for shape in draft["shapes"]]
    assert len(set(labels)) == len(labels) == 3
    assert set(labels) == {
        candidate["slot_id"]
        for candidate in evidence["candidates"]
        if candidate["polygon"]
    }
    assert (output / draft["imagePath"]).resolve() == image
    assert draft["imageData"] is None
    assert not load_labelme_configuration(draft_path, image).is_approved

    repeated = tmp_path / "repeated"
    _propose(image, repeated, revision=3)
    revised = json.loads((repeated / "draft.json").read_bytes())
    assert [shape["label"] for shape in revised["shapes"]] == labels

    draft["shapes"][0]["points"][0][0] += 1.25
    draft["shapes"].pop(1)
    reviewed = output / "reviewed.json"
    reviewed.write_text(json.dumps(draft), encoding="utf-8")
    config = output / "approved.json"
    stale = _approve(image, reviewed, config, original_hash)
    assert stale.returncode == 2 and "reviewed SHA256" in stale.stderr
    assert not config.exists()

    reviewed_hash = sha256(reviewed.read_bytes()).hexdigest()
    previewed = _run(
        tmp_path,
        "labelme_configuration",
        "preview",
        "--annotation",
        reviewed,
        "--image",
        image,
        "--output",
        output / "reviewed.png",
    )
    assert previewed.returncode == 0, previewed.stderr
    assert reviewed_hash in previewed.stdout
    approved = _approve(image, reviewed, config, reviewed_hash)
    assert approved.returncode == 0, approved.stderr

    restored = load_configuration(config)
    assert restored.is_approved and restored.revision == 2
    height, width = geometry_row_image.shape[:2]
    projections = restored.projections_for_camera(
        _CAMERA, image_width=width, image_height=height
    )
    assert {p.parking_slot_id for p in projections} == {labels[0], labels[2]}
    first = next(p for p in projections if p.parking_slot_id == labels[0])
    x, y = draft["shapes"][0]["points"][0]
    assert first.polygon.vertices[0] == Point(x / (width - 1), y / (height - 1))

    verified = output / "verified"
    result = _run(
        tmp_path,
        "geometry_setup",
        "verified",
        "--image",
        image,
        "--configuration",
        config,
        "--output-dir",
        verified,
        "--camera-id",
        _CAMERA,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads((verified / "verified.json").read_bytes())["projections"]
    assert {item["slot_id"] for item in rows} == {labels[0], labels[2]}
    assert {item["status"] for item in rows} == {"operator_verified"}
    assert image.read_bytes() == reference


def test_unresolved_proposal_does_not_create_a_labelme_shape(tmp_path: Path) -> None:
    image = tmp_path / "blank.png"
    assert cv2.imwrite(str(image), np.zeros((120, 160, 3), dtype=np.uint8))
    output = tmp_path / "proposed"
    _propose(image, output)
    draft = json.loads((output / "draft.json").read_bytes())
    evidence = json.loads((output / "evidence.json").read_bytes())
    assert not draft["shapes"]
    assert [row["status"] for row in evidence["candidates"]] == ["unresolved"]
    assert evidence["candidates"][0]["polygon"] is None


def test_partial_row_is_visible_in_evidence_and_cannot_approve_an_empty_draft(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "open T row.png"
    image = np.full((340, 440, 3), 35, dtype=np.uint8)
    for x, y in ((60, 50), (140, 64), (230, 80), (330, 96)):
        cv2.line(image, (x, y), (x + 5, y + 155), (230, 230, 230), 5)
        cv2.line(image, (x - 20, y), (x + 20, y), (230, 230, 230), 5)
    assert cv2.imwrite(str(image_path), image)
    output = tmp_path / "review"
    _propose(image_path, output)
    draft_path = output / "draft.json"
    draft = json.loads(draft_path.read_bytes())
    evidence = json.loads((output / "evidence.json").read_bytes())
    assert draft["shapes"] == []
    assert evidence["summary"]["full_roi"] == 0
    assert evidence["summary"]["partial_hypotheses"] >= 2
    partials = [c for c in evidence["candidates"] if c["geometry_kind"] == "partial"]
    assert all(
        c["polygon"] is None and c["slot_id"] is None and c["missing_boundaries"]
        for c in partials
    )
    configuration = output / "approved.json"
    approved = _approve(
        image_path,
        draft_path,
        configuration,
        sha256(draft_path.read_bytes()).hexdigest(),
    )
    assert approved.returncode == 2
    assert not configuration.exists()

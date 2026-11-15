"""End-to-end LSD, background registration, export and operator persistence."""

import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

from parkvision.configuration import load_configuration
from parkvision.geometry_sequence_setup import propose_sequence
from parkvision.geometry_setup import propose_geometry
from test_geometry_setup import _approve, _run


def _sequence(tmp_path: Path) -> Path:
    image = np.full((480, 640, 3), 35, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:65] = mask[390:] = 255
    mask[:, :55] = mask[:, 610:] = 255
    texture = np.random.default_rng(41).integers(20, 120, image.shape, np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
    image[mask > 0] = texture[mask > 0]
    cv2.line(image, (65, 90), (595, 90), (230, 230, 230), 5)
    cv2.line(image, (60, 350), (605, 350), (230, 230, 230), 5)
    for x in (100, 230, 380, 530):
        cv2.line(image, (x, 90), (x, 350), (230, 230, 230), 5)
    for index, x in enumerate((530, 100)):
        frame = image.copy()
        cv2.rectangle(frame, (x - 25, 75), (x + 25, 365), (25, 25, 25), -1)
        assert cv2.imwrite(str(tmp_path / f"{index}.png"), frame)
    assert cv2.imwrite(str(tmp_path / "background.png"), mask)
    manifest = tmp_path / "sequence.json"
    manifest.write_text(
        json.dumps(
            {
                "role": "controlled_synthetic_not_CCTV",
                "background_mask": "background.png",
                "frames": [
                    {
                        "image": f"{i}.png",
                        "source_id": "synthetic",
                        "camera_id": "fixed",
                    }
                    for i in range(2)
                ],
            }
        )
    )
    return manifest


def _kwargs():
    return dict(
        parking_id="synthetic",
        parking_name="Test only",
        camera_id="fixed",
        camera_name="Synthetic",
        revision=2,
    )


def test_real_lsd_and_registration_recover_third_synthetic_bay(tmp_path):
    manifest = _sequence(tmp_path)
    single, multi = tmp_path / "single", tmp_path / "multi"
    assert propose_geometry(tmp_path / "0.png", single, **_kwargs()) == (2, 0)
    result = _run(
        tmp_path,
        "geometry_setup",
        "propose-sequence",
        "--manifest",
        manifest,
        "--output-dir",
        multi,
        "--parking-id",
        "synthetic",
        "--parking-name",
        "Test only",
        "--camera-id",
        "fixed",
        "--camera-name",
        "Synthetic",
        "--revision",
        "2",
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads((multi / "evidence.json").read_bytes())
    assert evidence["summary"]["full_roi"] == 3
    assert all(f["accepted"] for f in evidence["multi_frame"]["frames"])
    assert not evidence["multi_frame"]["conflicts"]
    assert all(
        c["observation_ids"] for c in evidence["multi_frame"]["candidate_sources"]
    )
    assert (multi / "sequence-preview.png").is_file()
    first = json.loads((single / "draft.json").read_bytes())["shapes"]
    after = json.loads((multi / "draft.json").read_bytes())["shapes"]
    assert all(shape in after for shape in first)
    # The reviewed labels are physical identities; fusion never approves them.
    after[0]["label"] = "physical-stall-A"
    draft = json.loads((multi / "draft.json").read_bytes())
    draft["shapes"] = after
    reviewed = multi / "reviewed.json"
    reviewed.write_text(json.dumps(draft))
    config = multi / "approved-test-only.json"
    approval = _approve(
        tmp_path / "0.png", reviewed, config, sha256(reviewed.read_bytes()).hexdigest()
    )
    assert approval.returncode == 0, approval.stderr
    restored = load_configuration(config)
    assert restored.is_approved and restored.revision == 2
    assert "physical-stall-A" in {p.parking_slot_id for p in restored.projections}
    verified = _run(
        tmp_path,
        "geometry_setup",
        "verified",
        "--image",
        tmp_path / "0.png",
        "--configuration",
        config,
        "--output-dir",
        multi / "verified",
    )
    assert verified.returncode == 0, verified.stderr
    assert "physical-stall-A" in (multi / "verified/verified.json").read_text()


def test_repeated_path_is_a_duplicate_and_export_cannot_overwrite_review(tmp_path):
    manifest = _sequence(tmp_path)
    data = json.loads(manifest.read_text())
    data["frames"][1] = data["frames"][0]
    manifest.write_text(json.dumps(data))
    output = tmp_path / "review"
    assert propose_sequence(manifest, output, **_kwargs()) == (2, 0)
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["multi_frame"]["frames"][1]["reason"] == "duplicate_pixels"
    before = (output / "draft.json").read_bytes()
    with pytest.raises(FileExistsError):
        propose_sequence(manifest, output, **_kwargs())
    assert (output / "draft.json").read_bytes() == before


@pytest.mark.parametrize("frames", [[], [None], [{"image": "0.png"}]])
def test_manifest_cannot_silently_assign_a_shared_camera(tmp_path, frames):
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"frames": frames}))
    with pytest.raises(ValueError):
        propose_sequence(manifest, tmp_path / "review", **_kwargs())
    assert not (tmp_path / "review").exists()

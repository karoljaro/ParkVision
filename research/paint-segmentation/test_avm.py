"""Synthetic AVM mapping, archive validation, immutability and transfer tests."""

import zipfile

import cv2
import numpy as np
import pytest

import avm
from prepare import read_json, sha256, write_json


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "avm.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        for i in (1, 2):
            sample_id = f"{i:08}"
            image = np.full((160, 320, 3), 60 * i, np.uint8)
            mask = np.zeros_like(image)
            # OpenCV BGR: free, marker, vehicle, other, ego, unknown.
            for x, bgr in enumerate(
                [
                    (255, 0, 0),
                    (255, 255, 255),
                    (0, 0, 255),
                    (0, 255, 0),
                    (0, 0, 0),
                    (0, 255, 255),
                ]
            ):
                mask[:, x * 50 : (x + 1) * 50] = bgr
            for family, ext, value in [("images", "jpg", image), ("gt", "png", mask)]:
                ok, encoded = cv2.imencode(f".{ext}", value)
                assert ok
                zipped.writestr(f"{family}/{sample_id}.{ext}", encoded.tobytes())
            zipped.writestr(f"annotations/{sample_id}.yml", "%YAML:1.0\n")
        zipped.writestr(
            "info.txt", "/images/00000001.jpg 0 0 0\r/images/00000002.jpg 1 0 0\r"
        )
        for i, split in enumerate(("train", "test"), 1):
            zipped.writestr(f"{split}_db.txt", f"/images/{i:08}.jpg /gt/{i:08}.png\r")
        for script in ("split_db.py", "show_db_info.py"):
            zipped.writestr(
                script, "raise RuntimeError('must not execute publisher scripts')"
            )
    provenance = tmp_path / "source.json"
    write_json(
        provenance, {"archive_sha256": sha256(path), "purpose": "synthetic test"}
    )
    return path, provenance


def test_exact_palette_unknown_ignore_and_no_curb():
    colors = np.array(
        [list(avm.RGB_CLASSES) + [(255, 255, 0), (254, 255, 255)]], np.uint8
    )
    mapped = avm.map_mask(colors)
    assert mapped.tolist() == [[0, 0, 1, 255, 255, 255, 255]]
    assert 2 not in mapped
    with pytest.raises(ValueError, match="RGB"):
        avm.map_mask(np.zeros((160, 320), np.uint8))


def test_archive_roundtrip_and_frozen_files(archive, tmp_path):
    zipped, source = archive
    original_hash = sha256(zipped)
    root = tmp_path / "imported"
    report = avm.import_archive(zipped, root, source, count=2)
    assert report["images"] == 2 and report["curb_pixels"] == 0
    assert sum(report["pixel_counts"].values()) == 2 * 160 * 320
    assert report["unknown_color_frame_counts"] == {"(255, 255, 0)": 2}
    dataset = avm.verify_dataset(root)
    assert {s["role"] for s in dataset["samples"]} == {"external_train"}
    with zipfile.ZipFile(zipped) as z:
        assert (root / dataset["samples"][0]["source_mask"]).read_bytes() == z.read(
            "gt/00000001.png"
        )
    assert sha256(zipped) == original_hash
    with pytest.raises(ValueError, match="NEW directory"):
        avm.import_archive(zipped, root, source, count=2)
    path = root / dataset["samples"][0]["mask"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        avm.verify_dataset(root)


def test_wrong_count_hash_and_dimensions_are_rejected(archive, tmp_path):
    zipped, source = archive
    with pytest.raises(ValueError, match="count"):
        avm.import_archive(zipped, tmp_path / "bad-count", source, count=3)
    data = read_json(source)
    data["archive_sha256"] = "bad"
    write_json(source, data)
    with pytest.raises(ValueError, match="hash"):
        avm.import_archive(zipped, tmp_path / "bad-hash", source, count=2)
    ok, image = cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))
    assert ok
    with pytest.raises(ValueError, match="320x160"):
        avm.decode(image.tobytes())


def test_binary_loss_ignore_and_feature_transfer(tmp_path):
    torch = pytest.importorskip("torch")
    import train
    from torch import nn

    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[:, 1] = 1
    target[:, 3] = 255
    loss = train.training_loss(logits, target, torch.ones(2))
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(logits.grad[:, :, 3]) == 0

    binary, destination = nn.Module(), nn.Module()
    for m, classes in ((binary, 2), (destination, 3)):
        m.encoder = nn.Conv2d(3, 4, 1)
        m.decoder = nn.Conv2d(4, 4, 1)
        m.segmentation_head = nn.Conv2d(4, classes, 1)
    old_head = destination.segmentation_head.weight.detach().clone()
    checkpoint = tmp_path / "source.pt"
    state = {
        "epoch": 3,
        "model": binary.state_dict(),
        "config": {"schema": "parkvision-avm-pretraining-v1", "classes": avm.CLASSES},
    }
    torch.save(state, checkpoint)
    info = train.transfer_avm_features(destination, checkpoint, sha256(checkpoint))
    assert info["source_epoch"] == 3
    assert torch.equal(destination.encoder.weight, binary.encoder.weight)
    assert torch.equal(destination.decoder.weight, binary.decoder.weight)
    assert torch.equal(destination.segmentation_head.weight, old_head)
    with pytest.raises(ValueError, match="hash"):
        train.transfer_avm_features(destination, checkpoint, "bad")
    state["config"]["schema"] = "overfit"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="binary AVM"):
        train.transfer_avm_features(destination, checkpoint, sha256(checkpoint))

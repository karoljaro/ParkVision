"""Binary logical view, semantic transfer, ignored gradients and full output test."""

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import binary_paint as binary  # noqa: E402
import binary_views as views  # noqa: E402
import train  # noqa: E402
from torch import nn  # noqa: E402
from prepare import read_json, sha256, write_json  # noqa: E402

pytest_plugins = ("test_train",)


class Tiny(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.segmentation_head = nn.Conv2d(4, classes, 1)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


@pytest.fixture
def source(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Tiny()
    model.encoder[1].running_mean.fill_(0.1)
    path = tmp_path / "avm.pt"
    state = {
        "epoch": 3,
        "model": model.state_dict(),
        "optimizer": {"old": True},
        "config": {
            "schema": "parkvision-avm-pretraining-v1",
            "architecture": "smp.Unet/resnet50",
            "classes": {"background": 0, "paint": 1, "ignore": 255},
            "provenance": {
                "mapping": {
                    "marker": "paint",
                    "vehicle": "background",
                    "free_space": "background",
                    "other": "ignore",
                    "ego": "ignore",
                    "unknown_colors": "ignore",
                }
            },
        },
    }
    torch.save(state, path)
    return path, state


def test_view_is_immutable_and_ignore_has_zero_gradient():
    original = np.array([[0, 1, 2, 255]], np.uint8)
    copy = original.copy()
    mapped = binary.binary_mask(original)
    assert mapped.tolist() == [[0, 1, 0, 255]]
    assert np.array_equal(original, copy) and not np.shares_memory(mapped, original)
    logits = torch.randn(1, 2, 1, 4, requires_grad=True)
    loss = train.training_loss(
        logits, torch.from_numpy(mapped[None]).long(), torch.ones(2)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[..., 3]) == 0
    assert torch.count_nonzero(logits.grad[..., 2]) > 0  # curb remains supervised
    with pytest.raises(ValueError):
        binary.binary_mask(np.array([[0, 3]], np.uint8))


def test_three_class_argmax_collapse_keeps_paint_metrics():
    # Paint wins with probability .4: applying a .5 threshold would be wrong.
    logits = torch.tensor(
        [[[0.3, 0.4, 0.3], [0.6, 0.2, 0.2], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]]
    )
    pred3 = logits.argmax(-1).numpy().astype(np.uint8)
    target3 = np.array([[1, 1, 2, 0]], np.uint8)
    before = train.metrics_from_confusion(
        train.confusion_matrix(
            torch.from_numpy(pred3).long(), torch.from_numpy(target3).long()
        )
    )["classes"]["paint"]
    matrix = binary.confusion(
        torch.from_numpy(binary.binary_mask(pred3)).long(),
        torch.from_numpy(binary.binary_mask(target3)).long(),
    )
    after = binary.paint_metrics(matrix)
    for key in ("iou", "dice", "precision", "recall", "tp", "fp", "fn"):
        assert after[key] == before[key]
    assert binary.binary_mask(pred3)[0, 0] == 1


def test_full_transfer_includes_binary_head_and_buffers(source):
    path, state = source
    model = Tiny()
    old_hash = sha256(path)
    info = binary.load_avm_model(model, path, old_hash)
    assert info["all_tensors_equal_source"]
    assert {r["name"] for r in info["tensor_audit"]} == set(state["model"])
    assert all(torch.equal(v, state["model"][k]) for k, v in model.state_dict().items())
    assert any(r["name"].startswith("segmentation_head.") for r in info["tensor_audit"])
    assert any(r["kind"] == "buffer" for r in info["tensor_audit"])
    assert sha256(path) == old_hash
    with pytest.raises(ValueError, match="SHA256"):
        binary.load_avm_model(model, path, "bad")


@pytest.mark.parametrize("fault", ("classes", "other", "head"))
def test_transfer_rejects_incompatible_semantics_or_head(source, fault):
    path, state = source
    if fault == "classes":
        state["config"]["classes"] = {"background": 1, "paint": 0, "ignore": 255}
    elif fault == "other":
        state["config"]["provenance"]["mapping"]["other"] = "background"
    else:
        state["model"]["segmentation_head.weight"] = torch.zeros(3, 4, 1, 1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="semantics|tensor"):
        binary.load_avm_model(Tiny(), path, sha256(path))


def test_binary_run_seals_test_and_saves_native_probabilities(
    frozen, source, tmp_path, monkeypatch
):
    path, _ = source
    monkeypatch.setattr(binary, "build_model", lambda classes=2: Tiny(classes))
    checked = train.checked_sample
    seen = []

    def guarded(sample, root):
        assert sample["split"] != "test"
        seen.append(sample["split"])
        return checked(sample, root)

    monkeypatch.setattr(train, "checked_sample", guarded)
    monkeypatch.setattr(views, "checked_sample", guarded)
    tiles, _ = binary.load_binary_split(frozen, "train", 64)
    counts = [sum(int(np.count_nonzero(t.mask == i)) for t in tiles) for i in range(2)]
    audit = binary.load_avm_model(Tiny(), path, sha256(path))
    write_json(tmp_path / "initialization-preflight.json", audit)
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "frozen_root": str(frozen),
            "checkpoint": str(path),
            "checkpoint_sha256": sha256(path),
            "code_sha256": {},
            "train_pixel_counts": counts,
            "frozen_lock_sha256": sha256(frozen / "lock.json"),
            "frozen_manifest_sha256": sha256(frozen / "dataset.json"),
            "settings": {
                "tile_size": 64,
                "epochs_limit": 2,
                "batch_size": 2,
                "seed": 42,
                "gradient_clip_norm": 5.0,
                "optimizer": {
                    "encoder_lr": 1e-4,
                    "decoder_head_lr": 1e-3,
                    "weight_decay": 1e-4,
                },
                "early_stopping": {"patience": 2, "min_delta": 0.001},
            },
        },
    )
    args = argparse.Namespace(protocol=protocol, output=tmp_path / "run", device="cpu")
    result = binary.run(args)
    assert (
        result["checkpoint_reload_predictions_identical"]
        and result["checkpoint_reload_probabilities_identical"]
    )
    assert not result["test_evaluated"] and set(seen) == {"train", "val"}
    assert result["validation"]["paint"]["scored_pixels"] == 3 * 11 * 29
    assert read_json(args.output / "config.json")["new_optimizer_initially_empty"]
    weights = np.array(counts, dtype=np.float32) ** -0.5
    weights /= weights.mean()
    assert read_json(args.output / "config.json")[
        "class_weights_from_UFPR_only"
    ] == pytest.approx(weights.tolist())
    p = np.load(args.output / "validation/val.paint-probability.npy")
    roi = np.load(args.output / "validation/val.roi-paint-probability.npy")
    assert p.shape == (64, 96) and p.dtype == np.float32
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert np.isnan(roi).any() and np.isfinite(roi).any()
    tile_p = np.load(
        next((args.output / "validation/tiles").glob("*.paint-probability.npy"))
    )
    assert np.array_equal(p[:, :64], tile_p)
    with pytest.raises(ValueError, match="sealed"):
        binary.load_binary_split(frozen, "test", 64)
    train.verify_freeze(frozen)
    with pytest.raises(ValueError, match="NEW directory"):
        binary.run(args)
    audit["all_tensors_equal_source"] = False
    write_json(tmp_path / "initialization-preflight.json", audit)
    args.output = tmp_path / "bad-preflight"
    with pytest.raises(ValueError, match="preflight"):
        binary.run(args)
    assert not args.output.exists()

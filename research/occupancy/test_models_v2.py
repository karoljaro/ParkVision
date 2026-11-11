"""Backend selection, checkpoint rejection and unchanged observation contracts."""

import json
from types import SimpleNamespace

import pytest
import torch

from research.occupancy import models, observe
from research.occupancy.replay import observation_from_document
from research.occupancy.training_v1.data import RECIPE
from research.occupancy.training_v1.model import ARCHITECTURE
from test_observe import FakeModels, inputs  # noqa: F401


def checkpoint():
    return {
        "architecture": ARCHITECTURE,
        "recipe": dict(RECIPE),
        "classes": ["free", "occupied"],
        "epoch": 13,
        "manifest_sha256": models.TRAINING_SHA256,
        "state_dict": {
            "features.0.0.weight": torch.zeros(96, 4, 4, 4),
            "classifier.2.weight": torch.zeros(2, 768),
            "classifier.2.bias": torch.zeros(2),
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "resnet50"),
        ("recipe", {**RECIPE, "context": 0}),
        ("recipe", {**RECIPE, "size": 128}),
        ("classes", ["occupied", "free"]),
        ("epoch", 12),
        ("manifest_sha256", "changed"),
    ],
)
def test_wrong_checkpoint_contract_is_rejected(field, value):
    data = checkpoint()
    models.validate_convnext_checkpoint(data)
    data[field] = value
    with pytest.raises(ValueError, match="contract"):
        models.validate_convnext_checkpoint(data)


def test_wrong_channels_and_missing_weights_are_rejected():
    data = checkpoint()
    data["state_dict"]["features.0.0.weight"] = torch.zeros(96, 3, 4, 4)
    with pytest.raises(ValueError, match="four input"):
        models.validate_convnext_checkpoint(data)
    del data["state_dict"]
    with pytest.raises(ValueError, match="four input"):
        models.validate_convnext_checkpoint(data)


@pytest.mark.parametrize("backend", models.MODEL_CHOICES)
def test_loader_checks_all_local_hashes_before_loading(tmp_path, monkeypatch, backend):
    asset = tmp_path / "weight"
    asset.write_bytes(b"bad weights")
    keys = ["detector_weights", "per_slot_weights"]
    if backend == "acpds-rcnn":
        keys.append("acpds_pooler")
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(dict.fromkeys(keys, "weight")))
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Deserialization before hash check")
    )
    with pytest.raises(ValueError, match="hash"):
        models.load_models(path, model=backend)


def test_asset_manifests_are_not_interchangeable(tmp_path):
    path = tmp_path / "assets.json"
    path.write_text(
        json.dumps(
            dict.fromkeys(
                ["detector_weights", "per_slot_weights", "acpds_pooler"], "unused"
            )
        )
    )
    with pytest.raises(ValueError, match="exactly"):
        models.load_assets(path, "convnext-roi4")
    with pytest.raises(ValueError, match="Unknown"):
        models.load_models(path, model="typo")


@pytest.mark.parametrize("backend", [None, *models.MODEL_CHOICES])
def test_cli_selects_backend_and_rejects_typos(backend):
    command = [
        "--configuration",
        "config",
        "--camera-id",
        "camera",
        "--image",
        "image",
        "--session-id",
        "s",
        "--max-age-seconds",
        "1",
        "--assets",
        "assets",
        "--output",
        "output",
    ]
    if backend:
        command.extend(["--model", backend])
    args = observe.build_parser().parse_args(command)
    assert args.model == (backend or "acpds-rcnn")
    with pytest.raises(SystemExit):
        observe.build_parser().parse_args(command + ["--model", "typo"])


@pytest.mark.parametrize("backend", models.MODEL_CHOICES)
def test_selection_keeps_v1_ids_timing_visibility_and_temporal_adapter(
    inputs, monkeypatch, backend  # noqa: F811
):
    calls = []

    def load(path, device, model):
        calls.append((path, device, model))
        return FakeModels()

    monkeypatch.setattr(observe, "load_models", load)
    inputs.model = backend
    result = observe.observe(inputs)
    assert calls == [(inputs.assets, "cpu", backend)]
    assert result["schema_version"] == 1
    assert result["provenance"]["selected_model"] == backend
    assert result["camera_id"] == inputs.camera_id
    assert result["captured_at"] == inputs.captured_at.isoformat()
    assert all(r["visibility"] == "unassessed" for r in result["observations"])
    temporal = observation_from_document(result, "slot-01", "smoke")
    assert temporal.state == "free" and temporal.visibility == "unassessed"


def test_default_factory_is_rcnn_and_no_fallback_on_convnext_error(
    inputs, monkeypatch  # noqa: F811
):
    calls = []

    def old(path, device):
        calls.append("rcnn")
        return FakeModels()

    def broken(path, device):
        calls.append("convnext")
        raise ValueError("Checkpoint hash mismatch")

    monkeypatch.setattr(models, "BaselineModels", old)
    monkeypatch.setattr(models, "ConvNeXtModels", broken)
    assert observe.observe(inputs)["status"] == "observed"
    inputs.model = "convnext-roi4"
    result = observe.observe(inputs)
    assert calls == ["rcnn", "convnext"]
    assert result["status"] == "inference_error"
    assert result["provenance"]["selected_model"] == "convnext-roi4"
    assert result["provenance"]["models"] is None
    assert all(r["state"] == "unknown" for r in result["observations"])


def test_shared_detector_and_roi_order_are_preserved():
    # Different numbers of vertices must not change physical slot order.
    from test_observe import projection

    projections = [
        projection([(0, 0), (0.3, 0), (0.3, 1), (0, 1)], "a"),
        projection([(0.5, 0), (1, 0), (0.75, 1)], "b"),
    ]
    from research.occupancy.training_v1.model import tensor_input
    from research.occupancy.training_v1.data import encode_slot
    import numpy as np

    rgb = np.zeros((80, 120, 3), np.uint8)
    rgb[:, :, 0] = np.arange(120, dtype=np.uint8)

    class Score:
        def __call__(self, batch):
            # Inspect the exact frozen transform, including RGB and target mask.
            for item, p in zip(batch, projections, strict=True):
                points = np.float32([(v.x, v.y) for v in p.polygon.vertices])
                points *= np.float32((119, 79))
                expected = tensor_input(encode_slot(rgb, points.tolist())[0])
                assert torch.equal(item, expected)
            return torch.tensor([[4.0, -4.0], [-4.0, 4.0]])

    adapter = object.__new__(models.ConvNeXtModels)
    adapter.torch, adapter.device, adapter.slot = torch, "cpu", Score()
    scores = adapter.slot_probabilities(rgb, None, projections)
    assert scores[0] < 0.1 and scores[1] > 0.9
    assert models.ConvNeXtModels.predict is models.BaselineModels.predict


def test_model_rejects_missing_state_even_with_valid_metadata():
    data = checkpoint()
    data["state_dict"]["classifier.2.bias"] = SimpleNamespace(shape=(3,))
    with pytest.raises(ValueError, match="two output"):
        models.validate_convnext_checkpoint(data)

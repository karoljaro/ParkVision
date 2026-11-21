"""Device selection, timing and optional real CUDA parity on synthetic input."""

from datetime import datetime, timezone
from os import environ
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision

from parkvision.capture.frame import Frame
from parkvision.configuration import load_configuration
from research.occupancy import models
from research.occupancy.rules import assign_vehicles, classify


def test_cpu_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CPU must not probe CUDA")
    )
    assert models.resolve_device("cpu") == ("cpu", "cpu_requested")


@pytest.mark.parametrize(
    "cuda_build,reason",
    [(None, "torch_without_cuda"), ("12.6", "cuda_not_visible")],
)
def test_unavailable_cuda_is_explicit(monkeypatch, cuda_build, reason):
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert models.resolve_device("auto") == ("cpu", reason)
    with pytest.raises(ValueError, match=reason):
        models.resolve_device("cuda")


@pytest.mark.parametrize("fail_probe", [False, True])
def test_cuda_checks_detector_operator_on_resolved_device(monkeypatch, fail_probe):
    synchronizations = []
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronizations.append)
    monkeypatch.setattr(
        torch, "tensor", lambda data, device: SimpleNamespace(device=device)
    )

    def nms(boxes, scores, threshold):
        assert boxes.device == scores.device == "cuda:1"
        if fail_probe:
            raise RuntimeError("CUDA operator disabled")
        return SimpleNamespace(device=boxes.device)

    monkeypatch.setattr(torchvision.ops, "nms", nms)
    if fail_probe:
        selected, reason = models.resolve_device("auto")
        assert selected == "cpu" and "CUDA operator disabled" in reason
        with pytest.raises(RuntimeError, match="CUDA operator disabled"):
            models.resolve_device("cuda")
        assert not synchronizations
    else:
        assert models.resolve_device("auto") == ("cuda:1", "cuda_probe_passed")
        assert models.resolve_device("cuda") == ("cuda:1", "cuda_probe_passed")
        assert synchronizations == ["cuda:1", "cuda:1"]


def test_mixed_component_devices_are_rejected():
    adapter = object.__new__(models.BaselineModels)
    adapter.device = "cpu"
    adapter.slot = torch.nn.Linear(2, 2)
    adapter.detector = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(RuntimeError, match="detector parameters and buffers"):
        adapter._validated_component_devices()


def test_profile_keeps_model_output_and_regular_prediction_unsynchronized():
    adapter = object.__new__(models.BaselineModels)
    adapter.torch, adapter.device = torch, "cpu"
    adapter.weights = SimpleNamespace(
        transforms=lambda: lambda tensor: tensor,
        meta={"categories": ["background", "car"]},
    )
    adapter.detector = lambda inputs: [
        {
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.75]),
        }
    ]
    adapter.slot_probabilities = lambda rgb, tensor, projections: [0.85]
    ticks = iter(range(5))
    adapter._profile_time = lambda: next(ticks)
    frame = SimpleNamespace(image=np.zeros((3, 3, 3), dtype=np.uint8))
    regular = adapter.predict(frame, [])
    timings = {}
    assert adapter.predict(frame, [], timings=timings) == regular
    assert timings == {
        "preparation_seconds": 1,
        "detector_seconds": 1,
        "per_slot_seconds": 1,
        "output_seconds": 1,
        "inference_seconds": 4,
    }
    with pytest.raises(StopIteration):
        next(ticks)


@pytest.mark.parametrize(
    "backend,asset_name",
    [("acpds-rcnn", "assets-rcnn.json"), ("convnext-roi4", "assets-convnext.json")],
)
def test_real_cuda_parity(backend, asset_name):
    directory = environ.get("PARKVISION_CUDA_PARITY_ASSETS_DIR")
    if not directory:
        pytest.skip("Set PARKVISION_CUDA_PARITY_ASSETS_DIR for actual model parity")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not exposed to this process")

    assets = Path(directory) / asset_name
    repo = Path(__file__).resolve().parents[2]
    config = load_configuration(repo / "apps/engine/examples/parking-setup-v1.json")
    projections = config.projections_for_camera(
        "camera-east", image_width=1280, image_height=720
    )
    image = np.full((720, 1280, 3), 72, dtype=np.uint8)
    image[130:134, 70:600] = 220
    image[130:470, 70:74] = 220
    image[130:470, 300:304] = 220
    image[130:470, 596:600] = 220
    frame = Frame(
        "camera-east", 0, datetime(2026, 11, 21, 12, tzinfo=timezone.utc), image
    )
    cpu = models.load_models(assets, device="cpu", model=backend)
    gpu = models.load_models(assets, device="cuda", model=backend)
    cpu_scores, cpu_detections = cpu.predict(frame, projections)
    timings = {}
    gpu_scores, gpu_detections = gpu.predict(frame, projections, timings=timings)
    assert gpu_scores == pytest.approx(cpu_scores, abs=0.002)
    for cpu_detection, gpu_detection in zip(
        cpu_detections, gpu_detections, strict=True
    ):
        assert gpu_detection["category"] == cpu_detection["category"]
        assert gpu_detection["score"] == pytest.approx(
            cpu_detection["score"], abs=0.002
        )
        assert gpu_detection["bbox"] == pytest.approx(cpu_detection["bbox"], abs=1)
    cpu_assign = assign_vehicles(projections, 1280, 720, cpu_detections)
    gpu_assign = assign_vehicles(projections, 1280, 720, gpu_detections)
    for p, cpu_score, gpu_score in zip(
        projections, cpu_scores, gpu_scores, strict=True
    ):
        slot = p.parking_slot_id
        assert classify(cpu_score, cpu_assign[slot]) == classify(
            gpu_score, gpu_assign[slot]
        )
    report = gpu.device_report()
    assert report["selected"].startswith("cuda:")
    assert set(report["component_devices"].values()) == {report["selected"]}
    assert report["cuda_memory"]["peak_allocated_bytes"] > 0
    assert timings["inference_seconds"] >= timings["detector_seconds"] > 0

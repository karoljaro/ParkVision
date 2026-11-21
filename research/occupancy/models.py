"""Explicit, local-only occupancy backends with pinned checkpoint identities."""

import importlib.util
import json
import sys
from collections import defaultdict
from hashlib import sha256
from itertools import chain
from time import perf_counter

SLOT_SHA256 = "96901e9003a3710bba964e8cea7dc2cfa90d2c099ff102c98b3f02240748ce52"
DETECTOR_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
POOLER_SHA256 = "258c7e457b6c7edc55f4343e204603c4dfe7cccf06b0e2e69efdef25be0625f1"
CONVNEXT_SHA256 = "5f5672cfdea28cdb53942675d748d5f9d6c3abda5bc7fc18abe209d1d1331a23"
TRAINING_SHA256 = "30e7adc215dbdbf9501b27ea3d554857716eb5ffdd3373087e08d25463c9ec74"
MODEL_CHOICES = ("acpds-rcnn", "convnext-roi4")
DEVICE_CHOICES = ("cpu", "cuda", "auto")


def resolve_device(requested):
    """Select a device usable by PyTorch and the shared TorchVision detector."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unknown occupancy device: {requested}")
    if requested == "cpu":
        return "cpu", "cpu_requested"

    import torch

    if torch.version.cuda is None:
        reason = "torch_without_cuda"
    elif not torch.cuda.is_available():
        reason = "cuda_not_visible"
    else:
        try:
            from torchvision.ops import nms

            device = f"cuda:{torch.cuda.current_device()}"
            # Tensor allocation alone can't establish that the detector has CUDA ops.
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
            scores = torch.tensor([1.0], device=device)
            if str(nms(boxes, scores, 0.5).device) != device:
                raise RuntimeError("TorchVision NMS did not return a CUDA tensor")
            torch.cuda.synchronize(device)
            return device, "cuda_probe_passed"
        except (ImportError, RuntimeError, OSError) as error:
            if requested == "cuda":
                raise RuntimeError(
                    f"CUDA detector runtime probe failed: {error}"
                ) from error
            return "cpu", f"cuda_detector_probe_failed: {error}"
    if requested == "cuda":
        raise ValueError(f"CUDA is unavailable in this environment ({reason})")
    return "cpu", reason


def file_hash(path):
    result = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_assets(path, model="acpds-rcnn"):
    if model not in MODEL_CHOICES:
        raise ValueError(f"Unknown occupancy model: {model}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "detector_weights": DETECTOR_SHA256,
        "per_slot_weights": SLOT_SHA256 if model == "acpds-rcnn" else CONVNEXT_SHA256,
    }
    if model == "acpds-rcnn":
        expected["acpds_pooler"] = POOLER_SHA256
    if not isinstance(data, dict) or set(data) != set(expected):
        raise ValueError(f"Assets for {model} must specify exactly {sorted(expected)}")
    paths = {}
    for key, digest in expected.items():
        value = data[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid asset path: {key}")
        paths[key] = (path.parent / value).resolve()
        if file_hash(paths[key]) != digest:
            raise ValueError(f"Unexpected {model} asset hash: {key}")
    return paths


def load_detector(path, device):
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    detector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return weights, detector.eval().to(device)


class BaselineModels:
    def __init__(self, assets_path, device="cpu"):
        paths = load_assets(assets_path)
        import torch
        import torchvision
        from torch import nn
        from torchvision.models import resnet50
        from torchvision.ops.misc import FrozenBatchNorm2d

        device_selection = self._configure_device(torch, device)

        module_name = "_parkvision_occupancy_acpds_pooling_v1"
        pooler = sys.modules.get(module_name)
        if pooler is None:
            spec = importlib.util.spec_from_file_location(
                module_name, paths["acpds_pooler"]
            )
            pooler = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = pooler
            spec.loader.exec_module(pooler)

        # The published RCNN uses this backbone and state-dict structure.
        # Load all parameters locally instead of downloading ImageNet weights.
        class SlotModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
                self.backbone.fc = nn.Linear(2048, 2)

            def forward(self, image, rois):
                crops = pooler.roi_pool(image, rois, 128, "square")
                return self.backbone(crops)

        self.slot = SlotModel()
        self.slot.load_state_dict(
            torch.load(paths["per_slot_weights"], map_location="cpu", weights_only=True)
        )
        self.slot.eval().to(self.device)
        self.weights, self.detector = load_detector(
            paths["detector_weights"], self.device
        )
        self.provenance = {
            "per_slot": {"name": "ACPDS RCNN_128_square_gopro", "sha256": SLOT_SHA256},
            "detector": {
                "name": "Faster R-CNN ResNet-50 FPN v2 COCO_V1",
                "sha256": DETECTOR_SHA256,
            },
            "pooler_sha256": POOLER_SHA256,
            "device": self.device,
            "device_selection": device_selection,
            "component_devices": self._validated_component_devices(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        }

    def _configure_device(self, torch, requested):
        self.torch = torch
        self.device, reason = resolve_device(requested)
        torch.set_num_threads(4)
        return {"requested": requested, "selected": self.device, "reason": reason}

    def _validated_component_devices(self):
        result = {}
        for name, model in (("per_slot", self.slot), ("detector", self.detector)):
            devices = {
                str(tensor.device)
                for tensor in chain(model.parameters(), model.buffers())
            }
            if devices != {self.device}:
                raise RuntimeError(
                    f"Expected {name} parameters and buffers on {self.device}, "
                    f"found {sorted(devices)}"
                )
            result[name] = self.device
        return result

    def synchronize(self):
        """Wait for work on this model's CUDA device when measuring performance."""
        if self.device.startswith("cuda:"):
            self.torch.cuda.synchronize(self.device)

    def device_report(self):
        memory = None
        if self.device.startswith("cuda:"):
            cuda = self.torch.cuda
            free, total = cuda.mem_get_info(self.device)
            memory = {
                "free_bytes": free,
                "total_bytes": total,
                "allocated_bytes": cuda.memory_allocated(self.device),
                "reserved_bytes": cuda.memory_reserved(self.device),
                "peak_allocated_bytes": cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": cuda.max_memory_reserved(self.device),
            }
        return {
            **self.provenance["device_selection"],
            "component_devices": self._validated_component_devices(),
            "cuda_memory": memory,
        }

    def _profile_time(self):
        self.synchronize()
        return perf_counter()

    def slot_probabilities(self, rgb, tensor, projections):
        torch = self.torch
        grouped = defaultdict(list)
        for index, projection in enumerate(projections):
            grouped[len(projection.polygon.vertices)].append(
                (
                    index,
                    [(v.x, v.y) for v in projection.polygon.vertices],
                )
            )
        probabilities = [None] * len(projections)
        with torch.inference_mode():
            normalized = tensor.float() / 255
            mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[
                :, None, None
            ]
            std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[:, None, None]
            prepared = (normalized - mean) / std
            for items in grouped.values():
                for start in range(0, len(items), 32):
                    batch = items[start : start + 32]
                    rois = torch.tensor(
                        [roi for _, roi in batch],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scores = self.slot(prepared, rois).softmax(1)[:, 1].tolist()
                    for (index, _), score in zip(batch, scores, strict=True):
                        probabilities[index] = score
        return probabilities

    def predict(self, frame, projections, *, timings=None):
        """Optionally time synchronized phases without changing model precision."""
        torch = self.torch
        if timings is not None:
            started = self._profile_time()
        # Frame is BGR; all backends receive RGB from the same decoded pixels.
        rgb = frame.image[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        with torch.inference_mode():
            detector_input = self.weights.transforms()(tensor)
            if timings is not None:
                prepared = self._profile_time()
            output = self.detector([detector_input])[0]
            if timings is not None:
                detected = self._profile_time()
            probabilities = self.slot_probabilities(rgb, tensor, projections)
            if timings is not None:
                classified = self._profile_time()
        detections = []
        for box, score, label in zip(
            output["boxes"].tolist(),
            output["scores"].tolist(),
            output["labels"].tolist(),
            strict=True,
        ):
            category = self.weights.meta["categories"][label]
            if score >= 0.60 and category in {"car", "bus", "truck", "motorcycle"}:
                detections.append({"bbox": box, "score": score, "category": category})
        if timings is not None:
            finished = self._profile_time()
            timings.update(
                preparation_seconds=prepared - started,
                detector_seconds=detected - prepared,
                per_slot_seconds=classified - detected,
                output_seconds=finished - classified,
                inference_seconds=finished - started,
            )
        return probabilities, detections


def validate_convnext_checkpoint(checkpoint):
    # Lazy imports keep the original observer/replay usable without Torch.
    from research.occupancy.training_v1.data import RECIPE
    from research.occupancy.training_v1.model import ARCHITECTURE

    expected = {
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "classes": ["free", "occupied"],
        "epoch": 13,
        "manifest_sha256": TRAINING_SHA256,
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("ConvNeXt checkpoint/preprocessing contract mismatch")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or any(
        key not in state or tuple(state[key].shape) != shape
        for key, shape in (
            ("features.0.0.weight", (96, 4, 4, 4)),
            ("classifier.2.weight", (2, 768)),
            ("classifier.2.bias", (2,)),
        )
    ):
        raise ValueError("ConvNeXt requires four input channels and two output classes")


class ConvNeXtModels(BaselineModels):
    """Frozen epoch 13; reuse training transforms and the baseline detector."""

    def __init__(self, assets_path, device="cpu"):
        paths = load_assets(assets_path, "convnext-roi4")
        import torch
        import torchvision

        from research.occupancy.training_v1.data import RECIPE
        from research.occupancy.training_v1.model import ARCHITECTURE, build_model

        device_selection = self._configure_device(torch, device)
        checkpoint = torch.load(
            paths["per_slot_weights"], map_location="cpu", weights_only=True
        )
        validate_convnext_checkpoint(checkpoint)
        self.slot = build_model()
        self.slot.load_state_dict(checkpoint["state_dict"], strict=True)
        self.slot.eval().to(self.device)
        self.weights, self.detector = load_detector(
            paths["detector_weights"], self.device
        )
        self.provenance = {
            "per_slot": {
                "name": "ConvNeXt Tiny ImageNet RGB+ROI epoch 13",
                "sha256": file_hash(paths["per_slot_weights"]),
                "architecture": ARCHITECTURE,
                "epoch": 13,
                "recipe": dict(RECIPE),
                "normalization": "ImageNet RGB; ROI mask 0/1",
                "classes": ["free", "occupied"],
                "training_manifest_sha256": TRAINING_SHA256,
            },
            "detector": {
                "name": "Faster R-CNN ResNet-50 FPN v2 COCO_V1",
                "sha256": DETECTOR_SHA256,
            },
            "device": self.device,
            "device_selection": device_selection,
            "component_devices": self._validated_component_devices(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        }

    def slot_probabilities(self, rgb, tensor, projections):
        import numpy as np

        from research.occupancy.training_v1.data import encode_slot
        from research.occupancy.training_v1.model import tensor_input

        torch = self.torch
        height, width = rgb.shape[:2]
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(projections), 32):
                inputs = []
                for projection in projections[start : start + 32]:
                    polygon = np.float32(
                        [(v.x, v.y) for v in projection.polygon.vertices]
                    ) * np.float32((width - 1, height - 1))
                    array, _ = encode_slot(rgb, polygon.tolist())
                    inputs.append(tensor_input(array))
                batch = torch.stack(inputs).to(self.device)
                probabilities.extend(self.slot(batch).softmax(1)[:, 1].tolist())
        return probabilities


def load_models(assets_path, device="cpu", model="acpds-rcnn"):
    if model == "acpds-rcnn":
        return BaselineModels(assets_path, device=device)
    if model == "convnext-roi4":
        return ConvNeXtModels(assets_path, device=device)
    raise ValueError(f"Unknown occupancy model: {model}")

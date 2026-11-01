"""Isolated, frozen-data U-Net experiment. Test data never enters model selection."""

from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from overfit import Tile, native_tiles, tensors
from prepare import (
    CLASSES,
    Image,
    check_leakage,
    labelme_mask,
    load_manifest,
    overlay,
    read_image,
    read_json,
    sha256,
    validate_mask,
    write_image,
    write_json,
)


def checked_sample(sample: dict, root: Path) -> tuple[Image, Image]:
    """Validate exported bytes AND their exact agreement with reviewed polygons."""
    paths = {
        key: (root / sample[key]).resolve() for key in ("image", "annotation", "mask")
    }
    for key, path in paths.items():
        if sha256(path) != sample[f"{key}_sha256"]:
            raise ValueError(f"{sample['id']}: {key} changed after export")
    annotation = read_json(paths["annotation"])
    if (paths["annotation"].parent / annotation["imagePath"]).resolve() != paths[
        "image"
    ]:
        raise ValueError("Annotation references another image")
    image = read_image(paths["image"])
    decoded = cv2.imread(str(paths["mask"]), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Cannot decode mask")
    mask = cast(Image, decoded)
    validate_mask(mask, image.shape[:2])
    if (sample["height"], sample["width"]) != mask.shape:
        raise ValueError("Exported dimensions differ from mask")
    if not np.array_equal(mask, labelme_mask(annotation, image)):
        raise ValueError("Mask does not match reviewed polygons")
    return image, mask


def freeze_dataset(dataset: Path, split_manifest: Path, output: Path) -> Path:
    """Copy all approved inputs, without changing a byte, to a NEW snapshot."""
    if output.exists():
        raise ValueError("Freeze output must be a NEW directory")
    exported = read_json(dataset)
    if exported.get("classes") != CLASSES:
        raise ValueError("Class mapping differs from importer")
    original = {s["id"]: s for s in load_manifest(dataset)}
    assigned = load_manifest(split_manifest)
    if set(original) != {s["id"] for s in assigned}:
        raise ValueError("Split proposal must contain exactly the exported IDs")
    if {s["split"] for s in assigned} != {"train", "val", "test"}:
        raise ValueError("Freeze needs nonempty train, val and test splits")
    check_leakage(assigned, split_manifest.parent)
    merged = []
    for assignment in assigned:
        source = original[assignment["id"]]
        for key in ("image", "annotation"):
            if (dataset.parent / source[key]).resolve() != (
                split_manifest.parent / assignment[key]
            ).resolve():
                raise ValueError("Split proposal references different inputs")
        if source["image_sha256"] != assignment["image_sha256"]:
            raise ValueError("Split proposal image hash differs")
        sample = {
            **source,
            **assignment,
            "mask": str((dataset.parent / source["mask"]).resolve()),
        }
        checked_sample(sample, dataset.parent)
        merged.append(sample)
    output.mkdir(parents=True)
    records = []
    for sample in merged:
        record = copy.deepcopy(sample)
        directory = output / "data" / sample["id"]
        directory.mkdir(parents=True)
        for key in ("image", "annotation", "mask"):
            source_path = (dataset.parent / sample[key]).resolve()
            destination = directory / source_path.name
            shutil.copyfile(source_path, destination)
            if sha256(destination) != sample[f"{key}_sha256"]:
                raise ValueError("Input changed during freeze")
            record[key] = str(destination.relative_to(output))
        record.pop("overlay", None)
        records.append(record)
    frozen = {
        "version": 1,
        "classes": CLASSES,
        "samples": records,
        "status": "frozen_whole_parking_split",
    }
    write_json(output / "dataset.json", frozen)
    shutil.copyfile(dataset, output / "source-dataset.json")
    shutil.copyfile(split_manifest, output / "source-split-manifest.json")
    files = [
        {"path": str(path.relative_to(output)), "sha256": sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_json(
        output / "lock.json",
        {
            "version": 1,
            "dataset_sha256": sha256(output / "dataset.json"),
            "source_dataset_sha256": sha256(dataset),
            "source_split_manifest_sha256": sha256(split_manifest),
            "files": files,
            "test_policy": (
                "Hash/integrity freeze only; no test training, inference, "
                "metrics or tuning"
            ),
        },
    )
    verify_freeze(output)
    return output / "dataset.json"


def verify_freeze(root: Path) -> dict:
    lock = read_json(root / "lock.json")
    if lock.get("version") != 1:
        raise ValueError("Unsupported lock version")
    for item in lock["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    if sha256(root / "dataset.json") != lock["dataset_sha256"]:
        raise ValueError("Frozen manifest changed")
    data = read_json(root / "dataset.json")
    if data["classes"] != CLASSES:
        raise ValueError("Frozen classes changed")
    # Recheck metadata without decoding held-out test images during training.
    samples = load_manifest(root / "dataset.json")
    owners: dict[tuple[str, str], str] = {}
    for sample in samples:
        for field in ("parking_id", "camera_id", "group_id"):
            key = field, sample[field]
            if owners.setdefault(key, sample["split"]) != sample["split"]:
                raise ValueError(f"Frozen split leakage: {field}")
    return data


def load_split(root: Path, split: str, size: int) -> tuple[list[Tile], list[dict]]:
    if split not in {"train", "val"}:
        raise ValueError("Test split is sealed; only train/val can be decoded")
    samples = [
        s for s in read_json(root / "dataset.json")["samples"] if s["split"] == split
    ]
    if not samples:
        raise ValueError(f"Empty {split} split")
    tiles = []
    for sample in samples:
        image, mask = checked_sample(sample, root)
        tiles.extend(native_tiles(sample["id"], image, mask, size))
    return tiles, samples


def augment(tile: Tile, rng: random.Random) -> Tile:
    """Paired D4 geometry without interpolation; photometry changes images only."""
    turns = rng.randrange(4)
    image = np.rot90(tile.image, turns)
    mask = np.rot90(tile.mask, turns)
    if rng.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-0.10, 0.10) * 255
    image = np.clip(
        (image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255
    )
    image = np.rint(image).astype(np.uint8)
    return Tile(tile.sample_id, tile.x, tile.y, image.copy(), mask.copy())


def confusion_matrix(prediction: Tensor, target: Tensor) -> Tensor:
    valid = target != 255
    return torch.bincount(target[valid] * 3 + prediction[valid], minlength=9).reshape(
        3, 3
    )


def training_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Weighted CE + foreground Dice on reviewed pixels, deterministic on CUDA.

    Avoid CUDA's nondeterministic nll_loss2d reduction. Selecting valid pixels
    before log-softmax/gather yields the same weighted CE and no ignore gradient.
    """
    valid = target != 255
    if not valid.any():
        raise ValueError("A training batch needs reviewed pixels")
    values = logits.permute(0, 2, 3, 1)[valid]
    labels = target[valid]
    pixel_weights = weights[labels]
    log_probability = values.log_softmax(1).gather(1, labels[:, None]).squeeze(1)
    ce = -(log_probability * pixel_weights).sum() / pixel_weights.sum()
    probabilities = values.softmax(1)
    truth = torch.nn.functional.one_hot(labels, logits.shape[1])
    counts = truth.sum(0)
    present = counts[1:] > 0
    if not present.any():
        return ce
    intersection = (probabilities * truth).sum(0)
    denominator = probabilities.sum(0) + counts
    dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
    return ce + (1 - dice[1:][present]).mean()


def metrics_from_confusion(matrix: Tensor) -> dict:
    matrix = matrix.to(torch.int64).cpu()
    by_class = {}
    for name, index in CLASSES.items():
        if index == 255:
            continue
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        union = support + predicted - tp
        by_class[name] = {
            "iou": tp / union if union else None,
            "dice": 2 * tp / (support + predicted) if support + predicted else None,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "support": support,
            "predicted_pixels": predicted,
            "tp": tp,
            "fp": predicted - tp,
            "fn": support - tp,
        }
    foreground = [by_class[name]["iou"] for name in ("paint", "curb")]
    present = [value for value in foreground if value is not None]
    return {
        "classes": by_class,
        "foreground_macro_iou": sum(present) / len(present) if present else None,
        "confusion_true_rows": matrix.tolist(),
        "scored_pixels": int(matrix.sum()),
        "undefined_metric_policy": "null for zero denominator; excluded from macro",
    }


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    reference: float = -1.0
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        if value > self.reference + self.min_delta:
            self.reference, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def build_model(pretrained: Path | None = None, classes: int = 3) -> nn.Module:
    import segmentation_models_pytorch as smp  # type: ignore[import-untyped]

    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=classes
    )
    if pretrained is not None:
        from torchvision.models import ResNet50_Weights  # type: ignore[import-untyped]

        info = read_json(pretrained)
        path = (pretrained.parent / info["file"]).resolve()
        if (
            info["url"] != ResNet50_Weights.IMAGENET1K_V2.url
            or sha256(path) != info["sha256"]
        ):
            raise ValueError("ImageNet weights source/hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = state["conv1.weight"].clone()
        model.encoder.load_state_dict(state, strict=True)
        if not torch.equal(model.encoder.conv1.weight.detach(), expected):
            raise ValueError("ImageNet initialization did not load")
    return cast(nn.Module, model)


def transfer_avm_features(
    model: nn.Module, checkpoint: Path, expected_hash: str
) -> dict:
    """Transfer binary AVM encoder/decoder only; retain a fresh three-class head."""
    if sha256(checkpoint) != expected_hash:
        raise ValueError("AVM transfer checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config"].get("schema") != "parkvision-avm-pretraining-v1":
        raise ValueError("Transfer requires a binary AVM pretraining checkpoint")
    if state["config"].get("classes") != {"background": 0, "paint": 1, "ignore": 255}:
        raise ValueError("Unexpected AVM checkpoint classes")
    source = state["model"]
    destination = model.state_dict()
    heads = [key for key in destination if key.startswith("segmentation_head.")]
    features = [key for key in destination if key.startswith(("encoder.", "decoder."))]
    if set(source) != set(destination) or not heads or not features:
        raise ValueError("Transfer architecture keys differ")
    original_head = {key: destination[key].clone() for key in heads}
    for key in features:
        if source[key].shape != destination[key].shape:
            raise ValueError(f"Transfer feature shape differs: {key}")
        destination[key] = source[key]
    model.load_state_dict(destination, strict=True)
    restored = model.state_dict()
    if any(not torch.equal(restored[k], source[k]) for k in features) or any(
        not torch.equal(restored[k], original_head[k]) for k in heads
    ):
        raise ValueError("Feature/head transfer verification failed")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": expected_hash,
        "source_epoch": state["epoch"],
        "source_config": state["config"],
        "feature_tensors_transferred": len(features),
        "head_tensors_retained": heads,
        "policy": (
            "encoder+decoder including buffers; fresh three-class head, "
            "no binary head or optimizer transfer"
        ),
    }


def evaluate(
    model: nn.Module,
    tiles: list[Tile],
    weights: Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, Tensor]:
    model.eval()
    images, targets = tensors(tiles)
    with torch.inference_mode():
        logits = torch.cat(
            [model(batch.to(device)).cpu() for batch in images.split(batch_size)]
        )
        prediction = logits.argmax(1)
        loss = float(training_loss(logits, targets, weights.cpu()))
    result = metrics_from_confusion(confusion_matrix(prediction, targets))
    result["loss"] = loss
    result["per_image"] = {}
    for sample_id in dict.fromkeys(t.sample_id for t in tiles):
        selected = [i for i, tile in enumerate(tiles) if tile.sample_id == sample_id]
        result["per_image"][sample_id] = metrics_from_confusion(
            confusion_matrix(prediction[selected], targets[selected])
        )
    return result, prediction


def save_predictions(
    output: Path, root: Path, tiles: list[Tile], samples: list[dict], prediction: Tensor
) -> None:
    output.mkdir()
    tiles_dir = output / "tiles"
    tiles_dir.mkdir()
    cards = []
    for sample in samples:
        image, target = checked_sample(sample, root)
        raw = np.full_like(target, 255)
        indices = [i for i, tile in enumerate(tiles) if tile.sample_id == sample["id"]]
        for i in indices:
            tile = tiles[i]
            pred = prediction[i].numpy().astype(np.uint8)
            h, w = target[
                tile.y : tile.y + pred.shape[0], tile.x : tile.x + pred.shape[1]
            ].shape
            raw[tile.y : tile.y + h, tile.x : tile.x + w] = pred[:h, :w]
            write_image(tiles_dir / f"{tile.name}.prediction.png", pred)
            roi = np.where(tile.mask != 255, pred, 255).astype(np.uint8)
            strip = np.hstack(
                [tile.image, overlay(tile.image, tile.mask), overlay(tile.image, roi)]
            )
            strip = cv2.resize(strip, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            write_image(tiles_dir / f"{tile.name}.comparison.png", cast(Image, strip))
        roi = np.where(target != 255, raw, 255).astype(np.uint8)
        error = image.copy()
        error[(target != 255) & (raw != target)] = (0, 0, 255)
        prefix = output / sample["id"]
        write_image(Path(f"{prefix}.prediction.png"), raw)
        write_image(Path(f"{prefix}.roi-prediction.png"), roi)
        write_image(Path(f"{prefix}.target.png"), target)
        write_image(Path(f"{prefix}.errors.png"), error)
        views = np.hstack([image, overlay(image, target), overlay(image, roi)])
        write_image(Path(f"{prefix}.comparison.png"), views)
        card = cv2.resize(views, (1200, 300), interpolation=cv2.INTER_AREA)
        header = np.full((45, 1200, 3), 24, np.uint8)
        cv2.putText(
            header,
            sample["id"],
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            header,
            "Original | human ROI labels | model on reviewed ROI (ignore excluded)",
            (10, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cards.append(np.vstack([header, card]))
    write_image(output / "overview.jpg", np.vstack(cards))
    write_json(
        output / "README.json",
        {
            "columns": [
                "original",
                "human target ROI",
                "model prediction on reviewed ROI",
            ],
            "colors": {"background": "blue", "paint": "magenta", "curb": "orange"},
            "errors": "red only on misclassified reviewed pixels",
            "prediction_png": (
                "native uint8; predictions across retained grid tiles, "
                "255 outside inferred tiles"
            ),
            "roi_prediction_png": "native uint8; 255 outside human-reviewed ROI",
            "tile_comparisons": (
                "every retained validation tile, original/target/prediction, "
                "2x nearest; no cherry-picking"
            ),
        },
    )


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def plot_history(output: Path, history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [h["epoch"] for h in history[1:]],
        [h["train_batch_loss"] for h in history[1:]],
        label="train augmented batches",
    )
    axes[0].plot(
        [h["epoch"] for h in history],
        [h["validation"]["loss"] for h in history],
        label="CNR validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="CE + foreground Dice loss")
    for name in ("paint", "curb"):
        axes[1].plot(
            [h["epoch"] for h in history],
            [h["validation"]["classes"][name]["iou"] for h in history],
            label=name,
        )
    axes[1].set(xlabel="Epoch", ylabel="CNR reviewed-ROI IoU", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "learning-curves.png", dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.min_delta < 0:
        raise ValueError(
            "Epochs, patience and batch size must be positive; delta nonnegative"
        )
    if args.output.exists():
        raise ValueError("Run output must be a NEW directory")
    frozen = verify_freeze(args.frozen)
    train_tiles, train_samples = load_split(args.frozen, "train", args.tile_size)
    val_tiles, val_samples = load_split(args.frozen, "val", args.tile_size)
    counts = torch.tensor(
        [
            sum(int(np.count_nonzero(t.mask == i)) for t in train_tiles)
            for i in range(3)
        ],
        dtype=torch.float32,
    )
    if (counts == 0).any():
        raise ValueError("Training needs reviewed pixels of every class")
    weights = counts.rsqrt()
    weights /= weights.mean()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable; do not silently change the execution device"
        )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(args.seed)
    transfer_path = getattr(args, "transfer_checkpoint", None)
    model = build_model(None if transfer_path else args.pretrained)
    initialization = None
    if transfer_path:
        initialization = transfer_avm_features(
            model, transfer_path, args.transfer_sha256
        )
    model = model.to(device)
    encoder = cast(Any, model).encoder
    encoder_ids = {id(p) for p in encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder.parameters()), "lr": 1e-4},
            {
                "params": [p for p in model.parameters() if id(p) not in encoder_ids],
                "lr": 1e-3,
            },
        ],
        weight_decay=1e-4,
    )
    args.output.mkdir(parents=True)
    config = {
        "schema": "parkvision-independent-unet-v1",
        "architecture": "smp.Unet/resnet50",
        "encoder_weights": "torchvision.ResNet50_Weights.IMAGENET1K_V2",
        "pretrained": read_json(args.pretrained),
        "overfit_checkpoint_loaded": False,
        "frozen_root": str(args.frozen.resolve()),
        "frozen_lock_sha256": sha256(args.frozen / "lock.json"),
        "frozen_manifest_sha256": sha256(args.frozen / "dataset.json"),
        "seed": args.seed,
        "device": str(device),
        "epochs_limit": args.epochs,
        "batch_size": args.batch_size,
        "tile_size": args.tile_size,
        "tiling": (
            "native disjoint grid, each labeled pixel once, zero-image and "
            "255-target padding; no resizing or CLAHE"
        ),
        "augmentation": (
            "train only: uniform rotations 0/90/180/270 + random horizontal flip; "
            "contrast .85..1.15, brightness -.10..+.10; "
            "paired labels have no interpolation"
        ),
        "preprocessing": "BGR to RGB /255; mean .485,.456,.406, std .229,.224,.225",
        "optimizer": {
            "name": "AdamW",
            "encoder_lr": 1e-4,
            "decoder_head_lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "encoder_batchnorm": (
            "running statistics frozen; convolution and affine weights fine-tuned"
        ),
        "loss": (
            "weighted CE(ignore_index=255) + Dice on batch-present foreground; "
            "ignore excluded from both"
        ),
        "class_weights_from_train_only": weights.tolist(),
        "train_pixel_counts": counts.long().tolist(),
        "selection": (
            "highest CNR pooled-pixel foreground macro IoU; "
            "ties keep earliest epoch; no epoch-0 checkpoint selection"
        ),
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
            "metric": "CNR foreground macro IoU",
            "checkpoint_best_uses_any_strict_improvement": True,
        },
        "train_ids": [s["id"] for s in train_samples],
        "validation_ids": [s["id"] for s in val_samples],
        "train_tiles": len(train_tiles),
        "validation_tiles": len(val_tiles),
        "test_policy": (
            "PUCPR only integrity-hashed; never decoded by training/evaluation "
            "loaders; no inference/metrics/tuning"
        ),
        "environment": {
            p: version(p)
            for p in (
                "torch",
                "torchvision",
                "segmentation-models-pytorch",
                "numpy",
                "opencv-python-headless",
            )
        },
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "deterministic_algorithms": True,
        "precision": "float32; no AMP",
        "splits": {s["id"]: s["split"] for s in frozen["samples"]},
    }
    write_json(args.output / "config.json", config)
    if initialization is not None:
        config["encoder_weights"] = (
            "ImageNet -> binary AVM pretraining -> UFPR fine-tuning"
        )
        config["initialization"] = initialization
        write_json(args.output / "config.json", config)
        write_json(args.output / "transfer.json", initialization)
    for name in ("train.py", "overfit.py", "prepare.py"):
        shutil.copyfile(Path(__file__).parent / name, args.output / name)
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (args.output / "environment.txt").write_text(environment.stdout)
    initial, _ = evaluate(model, val_tiles, weights, device, args.batch_size)
    history = [
        {
            "epoch": 0,
            "validation": initial,
            "note": (
                "Transferred AVM encoder/decoder, fresh three-class head"
                if initialization
                else "ImageNet encoder, random decoder, before optimization"
            ),
        }
    ]
    write_json(args.output / "history.json", history)
    stopping = EarlyStopping(args.patience, args.min_delta)
    best_score, best_epoch = -1.0, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        encoder.eval()
        order = torch.randperm(len(train_tiles)).tolist()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [
                augment(train_tiles[i], rng)
                for i in order[start : start + args.batch_size]
            ]
            images, targets = tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model(images.to(device)), targets.to(device), weights.to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = evaluate(model, val_tiles, weights, device, args.batch_size)
        score = validation["foreground_macro_iou"]
        if score is None:
            raise ValueError("Validation needs foreground support")
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        stop = stopping.update(score)
        history.append(
            {
                "epoch": epoch,
                "train_batch_loss": sum(losses) / len(losses),
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        write_json(args.output / "history.json", history)
        state = {
            "config": config,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "early_stopping": {
                "reference": stopping.reference,
                "bad_epochs": stopping.bad_epochs,
            },
            "torch_rng": torch.get_rng_state(),
            "augmentation_rng": rng.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(args.output / "last.pt", state)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        print(
            f"epoch={epoch} train_loss={sum(losses)/len(losses):.5f} "
            f"val_loss={validation['loss']:.5f} foreground_mIoU={score:.5f} "
            f"best_epoch={best_epoch} "
            f"patience={stopping.bad_epochs}/{args.patience}",
            flush=True,
        )
        if stop:
            break
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    final, prediction = evaluate(model, val_tiles, weights, device, args.batch_size)
    reloaded = build_model().to(device)
    reloaded.load_state_dict(best["model"])
    repeated, prediction_again = evaluate(
        reloaded, val_tiles, weights, device, args.batch_size
    )
    if not torch.equal(prediction, prediction_again) or final != repeated:
        raise ValueError("Checkpoint reload changed validation predictions")
    save_predictions(
        args.output / "validation", args.frozen, val_tiles, val_samples, prediction
    )
    plot_history(args.output, history)
    verify_freeze(args.frozen)
    report = {
        "purpose": (
            "experimental AVM-to-UFPR transfer; AVM scenes unknown, "
            "not an independent benchmark"
            if initialization
            else "experimental cross-parking validation, not production benchmark"
        ),
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "stop_reason": "early_stopping" if stop else "epoch_limit",
        "elapsed_seconds": time.monotonic() - started,
        "validation": final,
        "initial_validation": initial,
        "checkpoint_sha256": sha256(args.output / "best.pt"),
        "checkpoint_reload_predictions_identical": True,
        "frozen_inputs_unchanged": True,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    write_json(args.output / "result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("dataset", type=Path)
    freeze.add_argument("split_manifest", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--frozen", type=Path, required=True)
    train.add_argument("--pretrained", type=Path, required=True)
    train.add_argument("--transfer-checkpoint", type=Path)
    train.add_argument("--transfer-sha256")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-delta", type=float, default=0.001)
    train.add_argument("--tile-size", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            print(freeze_dataset(args.dataset, args.split_manifest, args.output))
        else:
            result = run(args)
            print(
                f"best_epoch={result['best_epoch']} CNR foreground_mIoU="
                f"{result['validation']['foreground_macro_iou']:.6f}"
            )
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

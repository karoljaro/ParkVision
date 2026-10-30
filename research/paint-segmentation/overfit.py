"""Technical U-Net memorization check on reviewed pilot pixels, never a benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

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


@dataclass
class Tile:
    sample_id: str
    x: int
    y: int
    image: np.ndarray
    mask: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.sample_id}-x{self.x}-y{self.y}"


def native_tiles(sample_id: str, image: np.ndarray, mask: np.ndarray, size: int):
    """Disjoint original-scale grid; retain every labeled pixel exactly once."""
    if size <= 0 or size % 32:
        raise ValueError("Tile size must be a positive multiple of 32")
    validate_mask(mask, image.shape[:2])
    height, width = mask.shape
    for y in range(0, height, size):
        for x in range(0, width, size):
            target = mask[y : y + size, x : x + size]
            if not np.any(target != 255):
                continue
            ih, iw = target.shape
            padded_image = np.zeros((size, size, 3), dtype=np.uint8)
            padded_mask = np.full((size, size), 255, dtype=np.uint8)
            padded_image[:ih, :iw] = image[y : y + ih, x : x + iw]
            padded_mask[:ih, :iw] = target
            yield Tile(sample_id, x, y, padded_image, padded_mask)


def load_tiles(dataset_path: Path, size: int) -> tuple[list[Tile], dict]:
    dataset = read_json(dataset_path)
    if dataset.get("classes") != CLASSES:
        raise ValueError("Dataset class mapping differs from the importer")
    samples = load_manifest(dataset_path)
    if any(sample["split"] != "pilot" for sample in samples):
        raise ValueError("This technical check accepts only split=pilot")
    check_leakage(samples, dataset_path.parent)
    tiles = []
    for sample in samples:
        image_path = (dataset_path.parent / sample["image"]).resolve()
        annotation_path = (dataset_path.parent / sample["annotation"]).resolve()
        mask_path = dataset_path.parent / sample["mask"]
        if sha256(annotation_path) != sample["annotation_sha256"]:
            raise ValueError(f"{sample['id']}: annotation changed after export")
        if sha256(mask_path) != sample["mask_sha256"]:
            raise ValueError(f"{sample['id']}: mask changed after export")
        annotation = read_json(annotation_path)
        if (annotation_path.parent / annotation["imagePath"]).resolve() != image_path:
            raise ValueError("Annotation references another source image")
        image = read_image(image_path)
        decoded = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Cannot decode mask: {mask_path}")
        # The runtime validator checks the dtype asserted for OpenCV's broad stub.
        mask = cast(Image, decoded)
        validate_mask(mask, image.shape[:2])
        if (sample["height"], sample["width"]) != mask.shape:
            raise ValueError("Exported metadata dimensions differ from mask")
        if not np.array_equal(mask, labelme_mask(annotation, image)):
            raise ValueError("Mask does not match the reviewed annotation")
        tiles.extend(native_tiles(sample["id"], image, mask, size))
    return tiles, dataset


def training_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Weighted CE + foreground Dice; ignore has zero loss and zero gradient."""
    valid = target != 255
    if not valid.any():
        raise ValueError("A training batch needs reviewed pixels")
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    probabilities = logits.softmax(dim=1) * valid.unsqueeze(1)
    truth = F.one_hot(target.masked_fill(~valid, 0), 3).permute(0, 3, 1, 2)
    truth = truth * valid.unsqueeze(1)
    sums = truth.sum(dim=(0, 2, 3))
    present = sums[1:] > 0
    if not present.any():
        return ce
    intersection = (probabilities * truth).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + sums
    dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
    return ce + (1 - dice[1:][present]).mean()


def tensors(tiles: list[Tile]) -> tuple[Tensor, Tensor]:
    rgb = np.stack([tile.image[:, :, ::-1] for tile in tiles]).copy()
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    targets = torch.from_numpy(np.stack([tile.mask for tile in tiles])).long()
    return (images - mean) / std, targets


def evaluate(
    model, images, targets, weights, device, batch_size
) -> tuple[dict, Tensor]:
    """Same fixed training tiles, in eval mode; no validation/test dataset."""
    model.eval()
    with torch.inference_mode():
        outputs = [model(batch.to(device)).cpu() for batch in images.split(batch_size)]
        logits = torch.cat(outputs)
        loss = float(training_loss(logits, targets, weights.cpu()))
        predicted = logits.argmax(dim=1)
        valid = targets != 255
        encoded = targets[valid] * 3 + predicted[valid]
        confusion = torch.bincount(encoded, minlength=9).reshape(3, 3)
        recall = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
        precision = confusion.diag() / confusion.sum(dim=0).clamp_min(1)
    return {
        "training_loss": loss,
        "training_pixel_recall": recall.tolist(),
        "training_pixel_precision": precision.tolist(),
        "training_confusion_true_rows": confusion.tolist(),
    }, predicted


def fit_passed(initial: dict, current: dict) -> bool:
    return current["training_loss"] <= initial["training_loss"] * 0.2 and all(
        value >= 0.9
        for key in ("training_pixel_recall", "training_pixel_precision")
        for value in current[key]
    )


def save_previews(output: Path, tiles: list[Tile], predictions: Tensor) -> None:
    directory = output / "previews"
    directory.mkdir()
    cards = []
    selected = set()
    for sample_id in dict.fromkeys(tile.sample_id for tile in tiles):
        indices = [i for i, tile in enumerate(tiles) if tile.sample_id == sample_id]
        for foreground in (1, 2):
            best = max(
                indices,
                key=lambda i: int(np.count_nonzero(tiles[i].mask == foreground)),
            )
            if np.any(tiles[best].mask == foreground):
                selected.add(best)
    for i, tile in enumerate(tiles):
        prediction = predictions[i].numpy().astype(np.uint8)
        # The raw prediction covers the entire tile. Only reviewed pixels are scored.
        write_image(directory / f"{tile.name}.prediction.png", prediction)
        write_image(directory / f"{tile.name}.target.png", tile.mask)
        scored_prediction = np.where(tile.mask != 255, prediction, 255).astype(np.uint8)
        views = [
            tile.image,
            overlay(tile.image, tile.mask),
            overlay(tile.image, scored_prediction),
        ]
        strip = cv2.resize(
            np.hstack(views), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST
        )
        card = np.full((strip.shape[0] + 62, strip.shape[1], 3), 24, np.uint8)
        card[42 : 42 + strip.shape[0]] = strip
        cv2.putText(
            card, tile.name, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
        )
        cv2.putText(
            card,
            "Original | user labels | model on LABELED pixels only",
            (8, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            card,
            "TRAINING examples - no generalization assessment",
            (8, card.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        write_image(directory / f"{tile.name}.comparison.png", card)
        if i in selected:
            cards.append(card)
    write_image(output / "training-examples.png", np.vstack(cards))


def run(args: argparse.Namespace) -> dict:
    if min(args.steps, args.batch_size, args.eval_every) <= 0 or args.lr <= 0:
        raise ValueError(
            "Steps, batch size, evaluation interval and LR must be positive"
        )
    if args.output.exists():
        raise ValueError("Output must be a NEW directory")
    tiles, dataset = load_tiles(args.dataset, args.tile_size)
    source_hash = sha256(args.dataset)
    images, targets = tensors(tiles)
    counts = torch.bincount(targets[targets != 255], minlength=3).float()
    if (counts == 0).any():
        raise ValueError(
            "This three-class check needs reviewed pixels of all three classes"
        )
    weights = counts.rsqrt()
    weights /= weights.mean()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable; run with GPU access or explicitly use --device cpu"
        )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    # SMP 0.6.0 does not provide stubs or a py.typed marker.
    import segmentation_models_pytorch as smp  # type: ignore[import-untyped]

    # Random initialization: no pretrained checkpoint or external training images.
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "purpose": "technical_training_overfit_only",
        "architecture": "smp.Unet",
        "encoder": "resnet50",
        "encoder_weights": None,
        "classes": CLASSES,
        "split": "pilot",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": source_hash,
        "seed": args.seed,
        "max_steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tile_size": args.tile_size,
        "tile_policy": "disjoint native grid, no resizing, padding target=255",
        "augmentation": None,
        "preprocessing": (
            "RGB /255; mean=[.485,.456,.406], std=[.229,.224,.225]; no CLAHE"
        ),
        "loss": (
            "weighted CE(ignore_index=255) + "
            "Dice of foreground classes present in batch"
        ),
        "class_weights": weights.tolist(),
        "supervised_pixels": counts.long().tolist(),
        "fit_criterion": (
            "same-tile eval loss <=20% initial AND "
            "precision AND recall >=90% for EACH class"
        ),
        "validation_or_test_metrics": None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "segmentation-models-pytorch",
                "numpy",
                "opencv-python-headless",
            )
        },
    }
    write_json(args.output / "config.json", config)
    write_json(
        args.output / "tiles.json",
        [
            {
                "id": t.name,
                "sample_id": t.sample_id,
                "x": t.x,
                "y": t.y,
                "size": args.tile_size,
                "split": "pilot",
                "pixels": {
                    name: int(np.count_nonzero(t.mask == value))
                    for name, value in CLASSES.items()
                },
            }
            for t in tiles
        ],
    )
    initial, _ = evaluate(model, images, targets, weights, device, args.batch_size)
    history = [{"step": 0, **initial}]
    print(json.dumps(history[-1]), flush=True)
    started = time.perf_counter()
    device_weights = weights.to(device)
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randperm(len(tiles), generator=generator)[: args.batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = model(images[indices].to(device))
        loss = training_loss(logits, targets[indices].to(device), device_weights)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            current, _ = evaluate(
                model, images, targets, weights, device, args.batch_size
            )
            history.append({"step": step, **current})
            write_json(args.output / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
            if fit_passed(initial, current):
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final, predictions = evaluate(
        model, images, targets, weights, device, args.batch_size
    )
    checkpoint = args.output / "unet-resnet50.pt"
    torch.save(
        {"model_state": model.cpu().state_dict(), "config": config, "step": step},
        checkpoint,
    )
    # Verify the saved artifact with a fresh model, not the in-memory weights.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=3
    ).to(device)
    restored.load_state_dict(saved["model_state"])
    restored_score, restored_predictions = evaluate(
        restored, images, targets, weights, device, args.batch_size
    )
    if not torch.equal(predictions, restored_predictions):
        raise ValueError("Reloaded checkpoint changed argmax predictions")
    # Detect edits while this bounded run was executing; never write source data.
    if source_hash != sha256(args.dataset):
        raise ValueError("Dataset metadata changed during training")
    load_tiles(args.dataset, args.tile_size)
    save_previews(args.output, tiles, predictions)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [h["step"] for h in history], [h["training_loss"] for h in history], marker="o"
    )
    ax.set(
        xlabel="Optimizer steps",
        ylabel="CE + foreground Dice loss",
        title="Same training tiles only - no validation/test",
    )
    fig.tight_layout()
    fig.savefig(args.output / "training-loss.png", dpi=150)
    plt.close(fig)
    report = {
        "purpose": config["purpose"],
        "fit_criterion_passed": fit_passed(initial, final),
        "images": len(dataset["samples"]),
        "tiles": len(tiles),
        "steps": step,
        "initial": initial,
        "final": final,
        "training_seconds": elapsed,
        "peak_allocated_gpu_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload_predictions_identical": True,
        "restored_training_loss": restored_score["training_loss"],
        "sources_unchanged": True,
        "validation_or_test_metrics": None,
        "note": (
            "Memorization of reviewed pilot pixels only. "
            "No generalization, model comparison, or detector integration."
        ),
    }
    write_json(args.output / "result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["fit_criterion_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

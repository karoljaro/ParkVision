"""Binary AVM pretraining with a fixed endpoint; no CNR or PUCPR selection."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from avm import CLASSES, verify_dataset
from overfit import Tile, tensors
from prepare import overlay, read_image, read_json, sha256, write_image, write_json
from train import build_model, save_checkpoint, training_loss


class Frames(Dataset):
    def __init__(self, root: Path, samples: list[dict], seed: int):
        self.root, self.samples = root, samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def tile(self, index: int, augment: bool) -> Tile:
        sample = self.samples[index]
        image = read_image(self.root / sample["image"])
        mask = cv2.imread(str(self.root / sample["mask"]), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError("AVM paired mask dimensions differ")
        if augment:
            # Rectangular native images: paired flips, no 90-degree shape changes.
            if self.rng.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if self.rng.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            image = np.rint(
                np.clip(
                    (image.astype(np.float32) - 127.5) * self.rng.uniform(0.85, 1.15)
                    + 127.5
                    + self.rng.uniform(-0.10, 0.10) * 255,
                    0,
                    255,
                )
            ).astype(np.uint8)
        return Tile(sample["id"], 0, 0, image.copy(), mask.copy())

    def __getitem__(self, index):
        images, targets = tensors([self.tile(index, True)])
        return images[0], targets[0]


def paint_metrics(matrix: torch.Tensor) -> dict:
    matrix = matrix.cpu().long()
    tp = int(matrix[1, 1])
    fp = int(matrix[0, 1])
    fn = int(matrix[1, 0])

    def ratio(a, b):
        return a / b if b else None

    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "confusion_true_rows": matrix.tolist(),
        "scored_pixels": int(matrix.sum()),
    }


def diagnostics(model, dataset, indices, device, output):
    """Fixed training examples only; never a validation/generalization estimate."""
    model.eval()
    tiles = [dataset.tile(i, False) for i in indices]
    images, targets = tensors(tiles)
    with torch.inference_mode():
        pred = torch.cat([model(b.to(device)).argmax(1).cpu() for b in images.split(4)])
    valid = targets != 255
    matrix = torch.bincount(targets[valid] * 2 + pred[valid], minlength=4).reshape(2, 2)
    output.mkdir()
    cards = []
    for tile, prediction in zip(tiles, pred):
        raw = prediction.numpy().astype(np.uint8)
        roi = np.where(tile.mask != 255, raw, 255).astype(np.uint8)
        write_image(output / f"{tile.sample_id}.prediction.png", raw)
        strip = np.hstack(
            [tile.image, overlay(tile.image, tile.mask), overlay(tile.image, roi)]
        )
        header = np.full((28, strip.shape[1], 3), 24, np.uint8)
        cv2.putText(
            header,
            f"AVM train {tile.sample_id}: original | GT | prediction (ignore excluded)",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        cards.append(np.vstack([header, strip]))
    write_image(output / "overview.jpg", np.vstack(cards))
    result = {
        "purpose": "training-set diagnostics only, not validation",
        "ids": [t.sample_id for t in tiles],
        "paint": paint_metrics(matrix),
    }
    write_json(output / "metrics.json", result)
    return result


def run(args):
    if args.output.exists():
        raise ValueError("Pretraining output must be a NEW directory")
    if min(args.epochs, args.batch_size) < 1:
        raise ValueError("Positive epochs and batch size required")
    data = verify_dataset(args.dataset)
    if not data["provenance"].get("research_use_basis_accepted"):
        raise ValueError("Document the actual academic-use basis before training")
    if {s["role"] for s in data["samples"]} != {"external_train"}:
        raise ValueError(
            "AVM pretraining must not create an independent validation split"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable; execution device must not change silently"
            )
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model = build_model(args.pretrained, classes=2).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-4},
            {
                "params": [p for p in model.parameters() if id(p) not in encoder_ids],
                "lr": 1e-3,
            },
        ],
        weight_decay=1e-4,
    )
    counts = data["report"]["pixel_counts"]
    weights = torch.tensor(
        [counts["background"], counts["paint"]], dtype=torch.float32
    ).rsqrt()
    weights /= weights.mean()
    dataset = Frames(args.dataset, data["samples"], args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    config = {
        "schema": "parkvision-avm-pretraining-v1",
        "classes": CLASSES,
        "architecture": "smp.Unet/resnet50",
        "encoder_weights": "torchvision.ResNet50_Weights.IMAGENET1K_V2",
        "pretrained": read_json(args.pretrained),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset / "dataset.json"),
        "dataset_lock_sha256": sha256(args.dataset / "lock.json"),
        "provenance": data["provenance"],
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames_per_epoch": len(dataset),
        "selection": (
            "Fixed last epoch, declared before CNR evaluation. No AVM "
            "validation or test metrics, no CNR/PUCPR used here."
        ),
        "optimizer": {
            "name": "AdamW",
            "encoder_lr": 1e-4,
            "decoder_head_lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "loss": "weighted CE + Dice on present paint; ignore=255 excluded from both",
        "class_weights": weights.tolist(),
        "pixel_counts": counts,
        "encoder_batchnorm": (
            "ImageNet running statistics frozen; weights and affine "
            "parameters trainable"
        ),
        "augmentation": (
            "paired H/V flips (includes 180 rotation), image-only contrast "
            ".85..1.15 and brightness -.10..+.10; no resizing, mask "
            "interpolation or CLAHE"
        ),
        "preprocessing": "native 320x160 BGR to RGB /255; ImageNet mean/std",
        "precision": "float32",
        "deterministic_algorithms": True,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "diagnostic_policy": (
            "8 evenly spaced training images; not a model selection metric"
        ),
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", config)
    for name in ("avm.py", "pretrain_avm.py", "train.py", "overfit.py", "prepare.py"):
        shutil.copyfile(Path(__file__).parent / name, args.output / name)
    indices = sorted(set(np.linspace(0, len(dataset) - 1, 8, dtype=int).tolist()))
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()
        loss_sum, frames, epoch_steps = 0.0, 0, 0
        for step, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model(images.to(device)), targets.to(device), weights.to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite AVM loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            frames += len(images)
            epoch_steps += 1
            if step % 25 == 0:
                print(
                    f"AVM epoch={epoch}/{args.epochs} batch={step}/{len(loader)} "
                    f"loss={loss_sum/frames:.5f} "
                    f"elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch,
                "train_batch_loss_frame_weighted": loss_sum / frames,
                "frames": frames,
                "steps": epoch_steps,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        write_json(args.output / "history.json", history)
        state = {
            "config": config,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "augmentation_rng": dataset.rng.getstate(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(args.output / f"epoch-{epoch:02}.pt", state)
        save_checkpoint(args.output / "last.pt", state)
        print(
            f"AVM epoch={epoch} done frames={frames} loss={loss_sum/frames:.5f}",
            flush=True,
        )
    # Reload the selected endpoint and verify diagnostic predictions round-trip.
    original = diagnostics(
        model, dataset, indices, device, args.output / "diagnostics-before-reload"
    )
    checkpoint = torch.load(
        args.output / "last.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    repeated = diagnostics(model, dataset, indices, device, args.output / "diagnostics")
    if original != repeated or any(
        sha256(p) != sha256(args.output / "diagnostics" / p.name)
        for p in (args.output / "diagnostics-before-reload").glob("*.prediction.png")
    ):
        raise ValueError("AVM endpoint reload changed predictions")
    verify_dataset(args.dataset)
    report = {
        "epochs_completed": args.epochs,
        "steps": sum(h["steps"] for h in history),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256(args.output / "last.pt"),
        "checkpoint_reload_identical": True,
        "diagnostics": repeated,
        "test_evaluated": False,
        "CNR_used": False,
        "PUCPR_used": False,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    write_json(args.output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()

"""One binary paint experiment, using an immutable logical view of Dataset A."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from overfit import Tile, tensors
from prepare import read_json, sha256, validate_mask, write_json
from pretrain_avm import paint_metrics
from train import (
    EarlyStopping,
    augment,
    build_model,
    load_split,
    save_checkpoint,
    training_loss,
    verify_freeze,
)

CLASSES = {"non-paint": 0, "paint": 1, "ignore": 255}
SCHEMA = "parkvision-binary-paint-v1"


def binary_mask(mask: np.ndarray) -> np.ndarray:
    """A new array: curb is a negative for paint, not invalid CV geometry."""
    validate_mask(mask, mask.shape)
    result = mask.copy()
    result[mask == 2] = 0
    return result


def load_binary_split(root: Path, split: str, size: int):
    tiles, samples = load_split(root, split, size)  # existing sealed-test guard
    return [
        Tile(t.sample_id, t.x, t.y, t.image, binary_mask(t.mask)) for t in tiles
    ], samples


def load_avm_model(model: nn.Module, checkpoint: Path, expected_hash: str) -> dict:
    if sha256(checkpoint) != expected_hash:
        raise ValueError("AVM checkpoint SHA256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    mapping = config.get("provenance", {}).get("mapping", {})
    if (
        config.get("schema") != "parkvision-avm-pretraining-v1"
        or config.get("architecture") != "smp.Unet/resnet50"
        or config.get("classes") != {"background": 0, "paint": 1, "ignore": 255}
        or mapping.get("marker") != "paint"
        or mapping.get("free_space") != "background"
        or not str(mapping.get("vehicle", "")).startswith("background")
        or any(
            mapping.get(key) != "ignore" for key in ("other", "ego", "unknown_colors")
        )
    ):
        raise ValueError("Unverified AVM binary output semantics")
    source, destination = state["model"], model.state_dict()
    if set(source) != set(destination) or any(
        source[key].shape != destination[key].shape
        or source[key].dtype != destination[key].dtype
        for key in destination
    ):
        raise ValueError("AVM binary tensor keys/shapes/dtypes differ")
    model.load_state_dict(source, strict=True)
    restored = model.state_dict()
    if any(not torch.equal(restored[key], source[key]) for key in source):
        raise ValueError("AVM full-model transfer failed")
    parameters = dict(model.named_parameters())
    return {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": expected_hash,
        "source_epoch": state["epoch"],
        "source_config": config,
        "scope": "all encoder, decoder, binary head and buffers; no optimizer",
        "output_semantics": {"0": "verified non-paint negatives", "1": "visible paint"},
        "all_tensors_equal_source": True,
        "tensor_audit": [
            {
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "kind": "parameter" if key in parameters else "buffer",
                "sha256": hashlib.sha256(
                    value.cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            for key, value in restored.items()
        ],
    }


def confusion(prediction: Tensor, target: Tensor) -> Tensor:
    valid = target != 255
    return torch.bincount(target[valid] * 2 + prediction[valid], minlength=4).reshape(
        2, 2
    )


def evaluate(model, tiles, weights, device, batch_size):
    model.eval()
    images, target = tensors(tiles)
    with torch.inference_mode():
        logits = torch.cat(
            [model(batch.to(device)).cpu() for batch in images.split(batch_size)]
        )
        prediction = logits.argmax(1)
        probability = logits.softmax(1)[:, 1]
        loss = float(training_loss(logits, target, weights.cpu()))
    result = {
        "paint": paint_metrics(confusion(prediction, target)),
        "loss": loss,
        "per_image": {},
    }
    for sample_id in dict.fromkeys(t.sample_id for t in tiles):
        indices = [i for i, t in enumerate(tiles) if t.sample_id == sample_id]
        result["per_image"][sample_id] = paint_metrics(
            confusion(prediction[indices], target[indices])
        )
    return result, prediction, probability


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("Binary run output must be a NEW directory")
    protocol = read_json(args.protocol)
    settings = protocol["settings"]
    if (
        min(
            settings["epochs_limit"],
            settings["batch_size"],
            settings["early_stopping"]["patience"],
        )
        < 1
        or settings["early_stopping"]["min_delta"] < 0
    ):
        raise ValueError(
            "Positive epochs/batch/patience and nonnegative delta required"
        )
    root = Path(protocol["frozen_root"])
    frozen = verify_freeze(root)
    if (
        sha256(root / "lock.json") != protocol["frozen_lock_sha256"]
        or sha256(root / "dataset.json") != protocol["frozen_manifest_sha256"]
    ):
        raise ValueError("Binary protocol frozen inputs changed")
    for name, digest in protocol["code_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"Protocol source changed: {name}")
    train_tiles, train_samples = load_binary_split(root, "train", settings["tile_size"])
    val_tiles, val_samples = load_binary_split(root, "val", settings["tile_size"])
    counts = torch.tensor(
        [
            sum(int(np.count_nonzero(t.mask == i)) for t in train_tiles)
            for i in range(2)
        ],
        dtype=torch.float32,
    )
    if (counts == 0).any():
        raise ValueError("Binary training requires both classes")
    weights = counts.rsqrt()
    weights /= weights.mean()
    if counts.long().tolist() != protocol["train_pixel_counts"]:
        raise ValueError("UFPR binary counts differ from protocol")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; no silent device fallback")
        torch.cuda.manual_seed_all(settings["seed"])
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(settings["seed"])
    model = build_model(classes=2)
    initialization = load_avm_model(
        model, Path(protocol["checkpoint"]), protocol["checkpoint_sha256"]
    )
    if initialization != read_json(
        args.protocol.parent / "initialization-preflight.json"
    ):
        raise ValueError("Full binary transfer differs from preflight")
    model = model.to(device)
    encoder = cast(Any, model).encoder
    encoder_ids = {id(p) for p in encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(encoder.parameters()),
                "lr": settings["optimizer"]["encoder_lr"],
            },
            {
                "params": [p for p in model.parameters() if id(p) not in encoder_ids],
                "lr": settings["optimizer"]["decoder_head_lr"],
            },
        ],
        weight_decay=settings["optimizer"]["weight_decay"],
    )
    if optimizer.state:
        raise ValueError("Binary optimizer must start empty")
    config = copy.deepcopy(protocol)
    config.update(
        {
            "schema": SCHEMA,
            "classes": CLASSES,
            "device": str(device),
            "initialization": initialization,
            "new_optimizer_initially_empty": True,
            "train_ids": [s["id"] for s in train_samples],
            "validation_ids": [s["id"] for s in val_samples],
            "train_tiles": len(train_tiles),
            "validation_tiles": len(val_tiles),
            "class_weights_from_UFPR_only": weights.tolist(),
            "splits": {s["id"]: s["split"] for s in frozen["samples"]},
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        }
    )
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "transfer.json", initialization)
    for name in (
        "binary_paint.py",
        "binary_views.py",
        "train.py",
        "pretrain_avm.py",
        "avm.py",
        "overfit.py",
        "prepare.py",
    ):
        shutil.copyfile(Path(__file__).parent / name, args.output / name)
    env = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (args.output / "environment.txt").write_text(env.stdout)
    initial, _, _ = evaluate(model, val_tiles, weights, device, settings["batch_size"])
    history = [
        {
            "epoch": 0,
            "validation": initial,
            "note": (
                "AVM initialization diagnostic, not eligible for checkpoint "
                "selection (same as previous runs)"
            ),
        }
    ]
    write_json(args.output / "history.json", history)
    stop_config = settings["early_stopping"]
    stopping = EarlyStopping(stop_config["patience"], stop_config["min_delta"])
    best_score, best_epoch = -1.0, 0
    started = time.monotonic()
    for epoch in range(1, settings["epochs_limit"] + 1):
        model.train()
        encoder.eval()
        order = torch.randperm(len(train_tiles)).tolist()
        losses = []
        for start in range(0, len(order), settings["batch_size"]):
            batch = [
                augment(train_tiles[i], rng)
                for i in order[start : start + settings["batch_size"]]
            ]
            images, targets = tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model(images.to(device)), targets.to(device), weights.to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite binary loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip_norm"]
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _, _ = evaluate(
            model, val_tiles, weights, device, settings["batch_size"]
        )
        score = validation["paint"]["iou"]
        if score is None:
            raise ValueError("Paint support required on validation ROI")
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
            "best_epoch": best_epoch,
            "best_score": best_score,
            "torch_rng": torch.get_rng_state(),
            "augmentation_rng": rng.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "early_stopping": {
                "reference": stopping.reference,
                "bad_epochs": stopping.bad_epochs,
            },
        }
        save_checkpoint(args.output / "last.pt", state)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        print(
            f"epoch={epoch} train_loss={sum(losses)/len(losses):.5f} "
            f"paint_IoU={score:.5f} best_epoch={best_epoch} "
            f"patience={stopping.bad_epochs}/{stopping.patience}",
            flush=True,
        )
        if stop:
            break
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    final, prediction, probability = evaluate(
        model, val_tiles, weights, device, settings["batch_size"]
    )
    reloaded = build_model(classes=2).to(device)
    reloaded.load_state_dict(best["model"], strict=True)
    repeated, prediction2, probability2 = evaluate(
        reloaded, val_tiles, weights, device, settings["batch_size"]
    )
    if (
        final != repeated
        or not torch.equal(prediction, prediction2)
        or not torch.equal(probability, probability2)
    ):
        raise ValueError("Binary checkpoint reload changed predictions/probabilities")
    from binary_views import save_predictions, plot_history

    save_predictions(
        args.output / "validation",
        root,
        val_tiles,
        val_samples,
        prediction,
        probability,
        reloaded,
        device,
        settings["batch_size"],
    )
    plot_history(args.output, history)
    verify_freeze(root)
    report = {
        "purpose": "Limited CNR validation, not an independent benchmark",
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "stop_reason": "early_stopping" if stop else "epoch_limit",
        "validation": final,
        "initial_validation": initial,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256(args.output / "best.pt"),
        "checkpoint_reload_predictions_identical": True,
        "checkpoint_reload_probabilities_identical": True,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    write_json(args.output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result = run(parser.parse_args())
    print(f"CNR paint IoU={result['validation']['paint']['iou']:.6f}")


if __name__ == "__main__":
    main()

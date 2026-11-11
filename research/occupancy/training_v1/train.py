"""Real fine-tuning on verified train/val only; no baseline or test inference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision

from .data import RECIPE, file_hash, local_path, verify, write_json
from .model import ARCHITECTURE, build_model, tensor_input


class SlotDataset(Dataset):
    def __init__(self, root, samples, augment=False):
        self.samples = samples
        self.augment = augment
        # Preload small arrays; no decoding repeated for every epoch.
        self.arrays = [
            np.load(local_path(root, s["array"]), allow_pickle=False) for s in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (
            tensor_input(self.arrays[index], self.augment),
            self.samples[index]["label"],
        )


def metrics(logits, labels):
    scores = logits.softmax(1)[:, 1]
    chosen = scores >= 0.5
    free = labels == 0
    occupied = ~free
    certain = (scores <= 0.1) | (scores >= 0.9)
    return {
        "source_accuracy": chosen.eq(labels).float().mean().item(),
        "source_free_recall_at_0_5": (~chosen[free]).float().mean().item(),
        "source_occupied_recall_at_0_5": chosen[occupied].float().mean().item(),
        "score_coverage_at_0_1_0_9": certain.float().mean().item(),
        "source_false_free_at_0_1": int(((scores <= 0.1) & occupied).sum()),
        "source_free_total": int(free.sum()),
        "source_occupied_total": int(occupied.sum()),
        "note": (
            "Source-label diagnostics, not verified visibility or deployment quality"
        ),
    }


def evaluate(model, loader, device):
    model.eval()
    outputs, labels = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            outputs.append(model(inputs.to(device)).cpu())
            labels.append(targets)
    logits, labels = torch.cat(outputs), torch.cat(labels)
    loss = nn.functional.cross_entropy(logits, labels).item()
    return loss, metrics(logits, labels)


def inherit_best_checkpoint(resume, output, manifest_sha256, best_loss, epoch):
    """Preserve the global best when a resumed run never improves validation."""
    source = resume.parent / "best.pt"
    if not source.is_file():
        raise ValueError("Resume needs the original sibling best.pt")
    best = torch.load(source, map_location="cpu", weights_only=True)
    if (
        best["manifest_sha256"] != manifest_sha256
        or best["architecture"] != ARCHITECTURE
        or best["recipe"] != RECIPE
        or best["best_loss"] != best_loss
        or best["epoch"] > epoch
    ):
        raise ValueError("Inherited best checkpoint does not match resume")
    shutil.copyfile(source, output / "best.pt")
    return {"path": str(source.resolve()), "sha256": file_hash(source)}


def run(config_path: Path, data_root: Path, output: Path, device="auto", resume=None):
    config = json.loads(config_path.read_text())
    for key in ("epochs", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("Run must include full-backbone fine-tuning")
    if config["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported architecture")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent fallback")
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify(data_root, manifest)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Keep checkpoints outside repository")
    output.mkdir(parents=True, exist_ok=False)
    code_files = ("__init__.py", "model.py", "train.py", "data.py", "acpds.py")
    for filename in code_files:
        (output / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    pretrained = Path(config["pretrained"])
    model = build_model(pretrained, config["pretrained_sha256"]).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": config["backbone_lr"]},
            {"params": model.classifier.parameters(), "lr": config["head_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    start_epoch, best_loss, inherited_best = 0, float("inf"), None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Resume dataset mismatch")
        if checkpoint["architecture"] != ARCHITECTURE or checkpoint["recipe"] != RECIPE:
            raise ValueError("Resume architecture/transform mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        inherited_best = inherit_best_checkpoint(
            resume, output, file_hash(manifest_path), best_loss, start_epoch
        )
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"] and device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_samples = [s for s in manifest["samples"] if s["split"] == "train"]
    val_samples = [s for s in manifest["samples"] if s["split"] == "val"]
    counts = Counter(s["label"] for s in train_samples)
    weights = torch.tensor(
        [len(train_samples) / (2 * counts[i]) for i in (0, 1)], device=device
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config["label_smoothing"]
    )
    generator = torch.Generator().manual_seed(seed)
    if resume:
        generator.set_state(checkpoint["loader_rng"])
    train_loader = DataLoader(
        SlotDataset(data_root, train_samples, True),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        SlotDataset(data_root, val_samples),
        batch_size=config["batch_size"],
        num_workers=0,
    )
    provenance = {
        "config": config,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_hash(manifest_path),
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "device": device,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "resume": str(resume) if resume else None,
        "resume_sha256": file_hash(resume) if resume else None,
        "inherited_best": inherited_best,
        "code_sha256": {f: file_hash(output / f) for f in code_files},
        "calibration": "none; fixed thresholds are research heuristics",
    }
    write_json(output / "run.json", provenance)
    print(json.dumps(provenance), flush=True)
    updates = 0
    for epoch in range(start_epoch, config["epochs"]):
        full = epoch >= config["warmup_epochs"]
        model.train()
        model.features.requires_grad_(full)
        if not full:
            model.features.eval()
        started = time.monotonic()
        total_loss, total = 0.0, 0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            total_loss += loss.item() * len(labels)
            total += len(labels)
            if step == 1 or step % 10 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "samples_seen": total,
                            "train_loss": total_loss / total,
                            "full_backbone": full,
                        }
                    ),
                    flush=True,
                )
        val_loss, values = evaluate(model, val_loader, device)
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "val_loss": val_loss,
            "full_backbone": full,
            "seconds": time.monotonic() - started,
            "metrics": values,
        }
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "architecture": ARCHITECTURE,
            "manifest_sha256": provenance["manifest_sha256"],
            "recipe": RECIPE,
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "config": config,
            "classes": ["free", "occupied"],
        }
        # Atomic epoch checkpoints; earlier runs are never reused as output.
        temporary = output / "last.partial.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(record), flush=True)
    if not updates:
        raise ValueError("No optimizer steps; this is not a training run")
    write_json(
        output / "checkpoints.json",
        {
            "optimizer_steps_this_run": updates,
            "checkpoints": [
                {"path": str((output / p).resolve()), "sha256": file_hash(output / p)}
                for p in ("best.pt", "last.pt")
            ],
            "quality_claim": "No comparison with frozen baseline performed",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.config, args.data, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()

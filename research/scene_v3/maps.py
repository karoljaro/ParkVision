"""Partial-channel U-Net learning of canopy and parking-region candidates."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F

from .data import (
    file_hash,
    load_annotation,
    mask_array,
    read_rows,
    source_group,
    write_json,
)
from .prepare import pick_families

ARCHITECTURE = "smp.Unet_resnet50_canopy_parking_region_v3"
CLASSES = ["tree_canopy", "annotated_parking_region_candidate"]
RECIPE = {
    "size": 512,
    "letterbox": True,
    "rgb_imagenet": True,
    "black_pixels": "ignore",
}


def prepare(audit_root, output, protected=()):
    output.mkdir(parents=True, exist_ok=False)
    (output / "targets").mkdir()
    audit = json.loads((audit_root / "audit.json").read_text())
    source = {Path(a["path"]).name: a["sha256"][:16] for a in audit["archives"]}
    aerial = source["Parking spaces and tree canopies.v4i.coco-segmentation.zip"]
    region = source["Parking lot line.v2i.coco-segmentation.zip"]
    rows, cache = read_rows(audit_root), {}
    groups = {"train_aerial": [], "val": [], "train_region": []}
    for row in rows:
        if row["archive"] not in {aerial, region} or row["protected_pucpr"]:
            continue
        image, _, _ = load_annotation(audit_root, row, cache)
        if row["archive"] == aerial:
            group, confidence = source_group(
                row["original_name"], image.get("extra", {}).get("user_tags", [])
            )
            if confidence == "unknown":
                continue
            groups[
                "train_aerial" if group == "aerial-city:Paderborn" else "val"
            ].append({**row, "group": group})
        else:
            groups["train_region"].append(
                {**row, "group": "parkinglane-entire-location-unmapped"}
            )
    candidates = [
        *(
            dict(r, split="train", supervision=[True, True])
            for r in groups["train_aerial"]
        ),
        *(
            dict(r, split="val", supervision=[True, True])
            for r in pick_families(groups["val"], audit_root, 100)
        ),
        *(
            dict(r, split="train", supervision=[False, True])
            for r in pick_families(groups["train_region"], audit_root, 2000)
        ),
    ]
    protected_files = [
        {"path": str(p.resolve()), "sha256": file_hash(p)} for p in protected
    ]
    protected_hashes = set()
    for reference in protected_files:
        doc = json.loads(Path(reference["path"]).read_text())
        for row in doc.get("samples", doc.get("frames", [])):
            protected_hashes.update(
                row[k]
                for k in ("sha256", "image_sha256")
                if isinstance(row.get(k), str)
            )
    selected, seen, rejected = [], set(), []
    for row in candidates:
        if row["pixel_sha256"] in seen or row["sha256"] in protected_hashes:
            continue
        seen.add(row["pixel_sha256"])
        _, names, annotations = load_annotation(audit_root, row, cache)
        masks = np.zeros((2, row["height"], row["width"]), np.uint8)
        invalid = False
        for annotation in annotations:
            name = names[annotation["category_id"]]
            channel = (
                0
                if name == "tree"
                else (
                    1
                    if name in {"parking space", "on-street parking", "parkinglane"}
                    else None
                )
            )
            if channel is not None:
                try:
                    masks[channel] |= mask_array(
                        annotation, row["height"], row["width"]
                    )
                except ValueError as error:
                    rejected.append(
                        {
                            "id": row["id"],
                            "annotation_id": annotation["id"],
                            "class": name,
                            "reason": str(error),
                        }
                    )
                    invalid = True
                    break
        if invalid:
            continue
        target = output / "targets" / (row["sha256"] + ".npz")
        np.savez_compressed(target, masks=masks)
        selected.append(
            {
                **row,
                "target": str(target.relative_to(output)),
                "target_sha256": file_hash(target),
                "occupancy_label": None,
                "training_eligible": True,
                "occupancy_training_eligible": False,
                "exporter_split": next(
                    r["split"] for r in rows if r["id"] == row["id"]
                ),
                "source_staging_role": row["staging_role"],
                "staging_role": "development_" + row["split"],
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "canopy_and_parking_region_candidates",
        "classes": CLASSES,
        "audit_root": str(audit_root.resolve()),
        "audit_sha256": file_hash(audit_root / "audit.json"),
        "samples": selected,
        "counts": dict(Counter(r["split"] + ":" + r["group"] for r in selected)),
        "protected_manifests": protected_files,
        "rejected_raster_empty_images": rejected,
        "semantics": (
            "Parking region can span multiple slots; not paint, not an "
            "individual ground footprint, not legal/free status. "
            "Other/unlabelled pixels are not confirmed nonparking."
        ),
        "supervision": (
            "Robot parking-region images do not supervise the canopy channel; "
            "black/masked pixels are ignored."
        ),
        "validation": (
            "11 parent aerial tiles from Herne; Paderborn and entire robot "
            "region location train. Independent final test unavailable."
        ),
        "frozen_tests_used": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_model(pretrained=None, expected_sha256=None):
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=2
    )
    if pretrained is not None:
        if file_hash(pretrained) != expected_sha256:
            raise ValueError("ImageNet initialization hash mismatch")
        weights = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(
            {k: v for k, v in weights.items() if not k.startswith("fc.")}, strict=True
        )
    return model


def prepare_input(rgb, masks=None):
    height, width = rgb.shape[:2]
    size = RECIPE["size"]
    scale = size / max(height, width)
    w, h = round(width * scale), round(height * scale)
    x, y = (size - w) // 2, (size - h) // 2
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (124, 116, 104)
    image[y : y + h, x : x + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    valid = np.zeros((size, size), bool)
    valid[y : y + h, x : x + w] = cv2.resize(
        (rgb.max(2) > 8).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    target = np.zeros((2, size, size), np.uint8)
    if masks is not None:
        for c in range(2):
            target[c, y : y + h, x : x + w] = cv2.resize(
                masks[c], (w, h), interpolation=cv2.INTER_NEAREST
            )
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    tensor = (
        tensor - torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    ) / torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (
        tensor,
        torch.from_numpy(target).float(),
        torch.from_numpy(valid),
        (x, y, w, h),
    )


def map_loss(logits, targets, valid, supervised):
    allowed = valid[:, None] & supervised[:, :, None, None]
    if not allowed.any():
        raise ValueError("No supervised pixels")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = bce[allowed].mean()
    probabilities = logits.sigmoid() * allowed
    truth = targets * allowed
    dims = (0, 2, 3)
    dice = (2 * (probabilities * truth).sum(dims) + 1) / (
        probabilities.sum(dims) + truth.sum(dims) + 1
    )
    present = supervised.any(0)
    return loss + (1 - dice[present]).mean()


def load_sample(root, manifest, row):
    path = Path(manifest["audit_root"]) / row["file"]
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    masks = np.load(root / row["target"], allow_pickle=False)["masks"]
    image, target, valid, _ = prepare_input(rgb, masks)
    return image, target, valid, torch.tensor(row["supervision"], dtype=torch.bool)


def evaluate(model, rows, root, manifest):
    model.eval()
    intersections = np.zeros(2)
    unions = np.zeros(2)
    with torch.inference_mode():
        for row in rows:
            image, target, valid, supervision = load_sample(root, manifest, row)
            prediction = model(image[None].cuda()).sigmoid()[0].cpu() >= 0.5
            allowed = valid[None] & supervision[:, None, None]
            intersections += (
                ((prediction & (target >= 0.5)) & allowed).sum((1, 2)).numpy()
            )
            unions += ((prediction | (target >= 0.5)) & allowed).sum((1, 2)).numpy()
    iou = (intersections / np.maximum(unions, 1)).tolist()
    return {
        "IoU": dict(zip(CLASSES, iou, strict=True)),
        "mean_IoU": sum(iou) / 2,
        "parent_images": len(rows),
    }


def train(manifest_path, pretrained, expected_sha256, output, epochs=10):
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for this training run")
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if manifest["classes"] != CLASSES:
        raise ValueError("Map class contract mismatch")
    groups, seen = {}, set()
    protected_hashes = set()
    for reference in manifest.get("protected_manifests", []):
        p = Path(reference["path"])
        if file_hash(p) != reference["sha256"]:
            raise ValueError("Protected manifest changed")
        doc = json.loads(p.read_text())
        for row in doc.get("samples", doc.get("frames", [])):
            protected_hashes.update(
                row[k]
                for k in ("sha256", "image_sha256")
                if isinstance(row.get(k), str)
            )
    for row in manifest["samples"]:
        if row["sha256"] in protected_hashes or row["protected_pucpr"]:
            raise ValueError("Protected map source")
        if (
            groups.setdefault(row["group"], row["split"]) != row["split"]
            or row["pixel_sha256"] in seen
        ):
            raise ValueError("Map group/pixel leakage")
        seen.add(row["pixel_sha256"])
        if (
            file_hash(Path(manifest["audit_root"]) / row["file"]) != row["sha256"]
            or file_hash(root / row["target"]) != row["target_sha256"]
        ):
            raise ValueError("Map training input changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-code").mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, output / "source-code" / p.name)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(43)
    random.seed(43)
    np.random.seed(43)
    model = build_model(pretrained, expected_sha256).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(43)
    training = [r for r in manifest["samples"] if r["split"] == "train"]
    validation = [r for r in manifest["samples"] if r["split"] == "val"]
    protocol = {
        "architecture": ARCHITECTURE,
        "classes": CLASSES,
        "recipe": RECIPE,
        "manifest_sha256": file_hash(manifest_path),
        "initialization_sha256": expected_sha256,
        "epochs": epochs,
        "batch_size": 4,
        "seed": 43,
        "selection": "mean validation IoU; fixed before training",
        "validation_scope": manifest["validation"],
        "gpu": torch.cuda.get_device_name(0),
        "frozen_tests_used": False,
    }
    write_json(output / "protocol.json", protocol)
    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        rng.shuffle(training)
        model.train()
        total = 0.0
        for offset in range(0, len(training), 4):
            batch = [
                load_sample(root, manifest, r) for r in training[offset : offset + 4]
            ]
            images, targets, valid, supervised = [
                torch.stack([sample[i] for sample in batch]).cuda() for i in range(4)
            ]
            if rng.random() < 0.5:
                images = images.flip(3)
                targets = targets.flip(3)
                valid = valid.flip(2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = map_loss(model(images), targets, valid, supervised)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite map loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        scores = evaluate(model, validation, root, manifest)
        row = {
            "epoch": epoch,
            "loss_sum": total,
            "validation": scores,
            "seconds": time.monotonic() - start,
        }
        history.append(row)
        checkpoint = {
            **protocol,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "validation": scores,
        }
        torch.save(checkpoint, output / "last.pt")
        if scores["mean_IoU"] > best:
            best = scores["mean_IoU"]
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    write_json(
        output / "result.json",
        {
            "best_sha256": file_hash(output / "best.pt"),
            "best_mean_IoU": best,
            "epochs_completed": epochs,
            "production_promoted": False,
            "history": history,
        },
    )


def predict(rgb, checkpoint, expected_sha256, device):
    if file_hash(checkpoint) != expected_sha256:
        raise ValueError("Map checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        saved.get("architecture") != ARCHITECTURE
        or saved.get("classes") != CLASSES
        or saved.get("recipe") != RECIPE
    ):
        raise ValueError("Map checkpoint contract mismatch")
    model = build_model()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().to(device)
    tensor, _, _, (x, y, w, h) = prepare_input(rgb)
    with torch.inference_mode():
        scores = model(tensor[None].to(device)).sigmoid()[0].cpu().numpy()
    return (
        np.stack(
            [
                cv2.resize(
                    channel[y : y + h, x : x + w],
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for channel in scores
            ]
        ),
        saved,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protected", type=Path, action="append", default=[])
    p = sub.add_parser("train")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.audit, args.output, args.protected)["counts"]))
    else:
        train(args.manifest, args.pretrained, args.sha256, args.output, args.epochs)


if __name__ == "__main__":
    main()

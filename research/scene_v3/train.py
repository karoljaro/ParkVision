"""Real mixed-supervision vehicle-instance training; no frozen test inputs."""

import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import torch

from .data import file_hash, load_annotation, mask_array, verify_manifest, write_json
from .model import ARCHITECTURE, RECIPE, build_model, mask_supervision


class SceneDataset:
    def __init__(self, manifest, split, supervision):
        self.root = Path(manifest["audit_root"])
        self.rows = [
            r
            for r in manifest["samples"]
            if r["split"] == split and r["supervision"] == supervision
        ]
        self.supervision, self.cache = supervision, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = cv2.cvtColor(cv2.imread(str(self.root / row["file"])), cv2.COLOR_BGR2RGB)
        _, _, annotations = load_annotation(self.root, row, self.cache)
        selected = [a for a in annotations if a["id"] in row["vehicle_annotation_ids"]]
        boxes, masks = [], []
        height, width = rgb.shape[:2]
        for annotation in selected:
            x, y, w, h = annotation["bbox"]
            box = [
                max(0.0, x),
                max(0.0, y),
                min(float(width), x + w),
                min(float(height), y + h),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Degenerate vehicle box")
            boxes.append(box)
            if self.supervision == "mask":
                masks.append(mask_array(annotation, height, width))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(index),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        if self.supervision == "mask":
            target["masks"] = (
                torch.from_numpy(np.stack(masks))
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            )
        box = target["boxes"]
        target["area"] = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255, target


def evaluation(model, dataset, device, output, name):
    model.eval()
    images, annotations, predictions, ann_id = [], [], [], 1
    enabled = dataset.supervision == "mask"
    with torch.inference_mode(), mask_supervision(model, enabled):
        for index in range(len(dataset)):
            image, target = dataset[index]
            height, width = image.shape[1:]
            images.append({"id": index, "height": height, "width": width})
            for j, box in enumerate(target["boxes"].tolist()):
                x1, y1, x2, y2 = box
                ann = {
                    "id": ann_id,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
                if enabled:
                    encoded = coco_mask.encode(
                        np.asfortranarray(target["masks"][j].numpy())
                    )
                    ann["segmentation"] = encoded
                    ann["area"] = int(coco_mask.area(encoded))
                annotations.append(ann)
                ann_id += 1
            prediction = model([image.to(device)])[0]
            for j, (box, score) in enumerate(
                zip(
                    prediction["boxes"].cpu().tolist(),
                    prediction["scores"].cpu().tolist(),
                    strict=True,
                )
            ):
                if score < 0.05:
                    continue
                x1, y1, x2, y2 = box
                item = {
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
                if enabled:
                    mask = (prediction["masks"][j, 0].cpu().numpy() >= 0.5).astype(
                        np.uint8
                    )
                    item["segmentation"] = coco_mask.encode(np.asfortranarray(mask))
                predictions.append(item)
    captured = io.StringIO()
    result = {}
    with contextlib.redirect_stdout(captured):
        truth = COCO()
        truth.dataset = {
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "vehicle_instance"}],
        }
        truth.createIndex()
        if not predictions:
            result = {
                "bbox_AP": 0.0,
                "bbox_AP50": 0.0,
                "segm_AP": 0.0 if enabled else None,
                "segm_AP50": 0.0 if enabled else None,
            }
        else:
            detected = truth.loadRes(predictions)
            for kind in (["bbox", "segm"] if enabled else ["bbox"]):
                evaluator = COCOeval(truth, detected, kind)
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                result[kind + "_AP"] = float(evaluator.stats[0])
                result[kind + "_AP50"] = float(evaluator.stats[1])
                result[kind + "_AR100"] = float(evaluator.stats[8])
    (output / (name + ".coco.txt")).write_text(captured.getvalue())
    return dict(
        result,
        images=len(dataset),
        source_families=len({r["original_name"] for r in dataset.rows}),
    )


def train(manifest_path, detector_path, output, epochs=6, device="cuda"):
    manifest = verify_manifest(manifest_path)
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("This training run requires the available CUDA GPU")
    output.mkdir(exist_ok=False, parents=True)
    (output / "source-code").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, output / "source-code" / path.name)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = build_model(detector_path).to(device)
    datasets = {
        kind: SceneDataset(manifest, "train", kind) for kind in ("mask", "bbox_only")
    }
    validation = {
        kind: SceneDataset(manifest, "val", kind) for kind in ("mask", "bbox_only")
    }
    if any(not len(d) for d in [*datasets.values(), *validation.values()]):
        raise ValueError(
            "Both real mask and bbox datasets require train/validation groups"
        )
    protocol = {
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "classes": ["background", "vehicle_instance"],
        "manifest_sha256": file_hash(manifest_path),
        "initialization_sha256": file_hash(detector_path),
        "epochs": epochs,
        "seed": 42,
        "batch_size": 2,
        "optimizer": "AdamW",
        "lr": 0.0001,
        "selection": (
            "mean of mask-validation segm_AP50 and video-validation bbox_AP50; "
            "fixed before training"
        ),
        "evaluation_scope": manifest["split_caveat"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "amp": True,
        "frozen_tests_used": False,
        "code_sha256": {
            p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(output / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,
        weight_decay=0.0001,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = random.Random(42)
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        tasks = []
        for kind, dataset in datasets.items():
            indices = list(range(len(dataset)))
            generator.shuffle(indices)
            tasks.extend((kind, indices[i : i + 2]) for i in range(0, len(indices), 2))
        generator.shuffle(tasks)
        sums = defaultdict(float)
        for step, (kind, indices) in enumerate(tasks, 1):
            model.train()
            # Frozen early feature parameters must not update BN running stats.
            for name, module in model.backbone.body.named_children():
                if name not in {"layer3", "layer4"}:
                    module.eval()
            batch = [datasets[kind][i] for i in indices]
            images = [im.to(device) for im, _ in batch]
            targets = [
                {k: v.to(device) for k, v in target.items()} for _, target in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with mask_supervision(model, kind == "mask"), torch.autocast(
                "cuda", dtype=torch.float16
            ):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                sums[kind + ":" + name] += float(value.detach())
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(tasks),
                            "loss": float(loss.detach()),
                        }
                    ),
                    flush=True,
                )
        scores = {
            kind: evaluation(model, data, device, output, f"epoch-{epoch:02d}-{kind}")
            for kind, data in validation.items()
        }
        metric = (scores["mask"]["segm_AP50"] + scores["bbox_only"]["bbox_AP50"]) / 2
        row = {
            "epoch": epoch,
            "selection_metric": metric,
            "validation": scores,
            "loss_sums": dict(sums),
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        checkpoint = {
            "architecture": ARCHITECTURE,
            "recipe": RECIPE,
            "classes": protocol["classes"],
            "manifest_sha256": protocol["manifest_sha256"],
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "validation": scores,
            "selection_metric": metric,
        }
        torch.save(checkpoint, output / "last.pt")
        if metric > best:
            best = metric
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    write_json(
        output / "result.json",
        {
            "best_sha256": file_hash(output / "best.pt"),
            "best_selection_metric": best,
            "epochs_completed": epochs,
            "history": history,
            "production_promoted": False,
        },
    )
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    train(args.manifest, args.detector, args.output, args.epochs)


if __name__ == "__main__":
    main()

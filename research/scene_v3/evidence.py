"""Auxiliary many-to-many instance/ROI evidence; never overwrite occupancy."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from research.occupancy import observe
from .data import file_hash
from .model import ARCHITECTURE, load_checkpoint


def overlaps(projections, masks, scores, width, height):
    """A vehicle may overlap zero, one or several slots; no unique owner rule."""
    roi_masks = {}
    for projection in projections:
        points = np.float32(
            [(v.x, v.y) for v in projection.polygon.vertices]
        ) * np.float32((width - 1, height - 1))
        canvas = np.zeros((height, width), np.uint8)
        cv2.fillPoly(canvas, [np.rint(points).astype(np.int32)], 1)
        roi_masks[projection.parking_slot_id] = canvas.astype(bool)
    instances, slot_rows = [], {key: [] for key in roi_masks}
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        if (
            mask.shape != (height, width)
            or not np.isfinite(mask).all()
            or not 0 <= score <= 1
        ):
            raise ValueError("Invalid scene mask or score")
        selected = mask >= 0.5
        area = int(selected.sum())
        if not area:
            continue
        support = []
        for slot_id, roi in roi_masks.items():
            intersection = int((selected & roi).sum())
            if not intersection:
                continue
            measurement = {
                "slot_id": slot_id,
                "intersection_pixels": intersection,
                "roi_fraction": intersection / int(roi.sum()),
                "instance_fraction": intersection / area,
            }
            support.append(measurement)
            slot_rows[slot_id].append({"instance_id": index, **measurement})
        instances.append(
            {
                "instance_id": index,
                "score": float(score),
                "mask_pixels": area,
                "slot_overlaps": support,
                "multi_slot_support": len(support) > 1,
                "interpretation": (
                    "image-plane overlap may be perspective, true multi-slot blockage, "
                    "or mask error; no forced owner"
                ),
            }
        )
    return instances, [
        {
            "slot_id": key,
            "instances": value,
            "visibility": "unassessed",
            "occupancy_override": None,
            "no_instance_does_not_establish_free": True,
        }
        for key, value in slot_rows.items()
    ]


def analyze(
    observation_path,
    configuration_path,
    checkpoint,
    expected_sha256,
    output,
    device="cuda",
    map_checkpoint=None,
    map_sha256=None,
):
    if bool(map_checkpoint) != bool(map_sha256):
        raise ValueError("Map checkpoint and hash must be supplied together")
    torch.set_num_threads(4)
    source = json.loads(observation_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("kind") != "experimental_offline_occupancy"
        or source["status"] != "observed"
    ):
        raise ValueError("Require a valid saved observer v1 frame")
    if file_hash(configuration_path) != source["configuration_sha256"]:
        raise ValueError("Source observation/configuration mismatch")
    image_path = Path(source["image"]["path"])
    if file_hash(image_path) != source["image"]["sha256"]:
        raise ValueError("Source frame changed")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    height, width = bgr.shape[:2]
    config = observe.load_configuration(configuration_path)
    projections = config.projections_for_camera(
        source["camera_id"], image_width=width, image_height=height
    )
    if {p.parking_slot_id for p in projections} != {
        r["slot_id"] for r in source["observations"]
    }:
        raise ValueError("Source physical slot IDs changed")
    model, metadata = load_checkpoint(checkpoint, expected_sha256, device)
    tensor = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
    with torch.inference_mode():
        prediction = model([tensor.to(device)])[0]
    keep = prediction["scores"] >= 0.6
    masks = prediction["masks"][keep, 0].cpu().numpy()
    scores = prediction["scores"][keep].cpu().tolist()
    instances, slots = overlaps(projections, masks, scores, width, height)
    source_slots = {row["slot_id"]: row for row in source["observations"]}
    for row in slots:
        row["source_visibility"] = source_slots[row["slot_id"]]["visibility"]
        row["source_state"] = source_slots[row["slot_id"]]["state"]
    directory = output.with_name(output.stem + "-masks")
    directory.mkdir(exist_ok=False)
    preview = bgr.copy()
    for index, mask in enumerate(masks):
        path = directory / f"instance-{index:03d}.png"
        cv2.imwrite(str(path), ((mask >= 0.5) * 255).astype(np.uint8))
        color = np.array(
            (
                (67 + index * 67) % 255,
                (220 + index * 31) % 255,
                (130 + index * 53) % 255,
            )
        )
        selected = mask >= 0.5
        preview[selected] = (preview[selected] * 0.6 + color * 0.4).astype(np.uint8)
    for projection in projections:
        pts = np.float32(
            [(v.x, v.y) for v in projection.polygon.vertices]
        ) * np.float32((width - 1, height - 1))
        cv2.polylines(preview, [np.rint(pts).astype(np.int32)], True, (0, 255, 255), 2)
    preview_path = directory / "preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    map_model = None
    if map_checkpoint is not None:
        from .maps import CLASSES, predict

        probabilities, map_metadata = predict(
            bgr[:, :, ::-1].copy(), map_checkpoint, map_sha256, device
        )
        for channel, name in zip(probabilities, CLASSES, strict=True):
            cv2.imwrite(
                str(directory / (name + ".png")),
                np.rint(channel * 255).astype(np.uint8),
            )
        for projection, row in zip(projections, slots, strict=True):
            points = np.float32(
                [(v.x, v.y) for v in projection.polygon.vertices]
            ) * np.float32((width - 1, height - 1))
            roi = np.zeros((height, width), np.uint8)
            cv2.fillPoly(roi, [np.rint(points).astype(np.int32)], 1)
            active = roi.astype(bool)
            row["scene_maps"] = {
                name: {
                    "mean_score": float(channel[active].mean()),
                    "fraction_above_0_5": float((channel[active] >= 0.5).mean()),
                }
                for name, channel in zip(CLASSES, probabilities, strict=True)
            }
            row["map_evidence_changes_visibility_or_parking_legality"] = False
        map_model = {
            "architecture": map_metadata["architecture"],
            "sha256": map_sha256,
            "epoch": map_metadata["epoch"],
            "classes": CLASSES,
            "scope": (
                "canopy and annotated region candidates; not readability or legal "
                "status"
            ),
        }
    result = {
        "schema_version": 1,
        "kind": "auxiliary_vehicle_instance_evidence",
        **{
            k: source[k]
            for k in (
                "parking_id",
                "camera_id",
                "configuration_revision",
                "configuration_sha256",
                "captured_at",
                "source_session",
                "sequence",
                "image",
            )
        },
        "source_observation_sha256": file_hash(observation_path),
        "model": {
            "architecture": ARCHITECTURE,
            "sha256": expected_sha256,
            "epoch": metadata["epoch"],
            "training_manifest_sha256": metadata["manifest_sha256"],
            "device": device,
        },
        "thresholds": {"instance_score": 0.6, "mask": 0.5},
        "mask_semantics": (
            "source vehicle-instance region; not certified visible pixels, "
            "ground contact, or legal slot footprint"
        ),
        "instances": instances,
        "slot_evidence": slots,
        "mask_directory": str(directory.resolve()),
        "original_occupancy_changed": False,
        "automatic_visibility_gate": False,
        "scene_map_model": map_model,
    }
    observe.save_result(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--map-checkpoint", type=Path)
    parser.add_argument("--map-sha256")
    args = parser.parse_args()
    result = analyze(
        args.observation,
        args.configuration,
        args.checkpoint,
        args.sha256,
        args.output,
        args.device,
        args.map_checkpoint,
        args.map_sha256,
    )
    print(
        json.dumps(
            {
                "instances": len(result["instances"]),
                "slots": len(result["slot_evidence"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

"""One frozen held-out comparison. This command cannot train or choose weights.

Core protocol and assets are snapshotted before inference. The official ACPDS
test is read only by this evaluation module, never admitted by training loaders.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from .comparison_metrics import methods_summary, summarize, visibility_gate
from .data import file_hash, local_path, write_json


def module_from_file(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_protocol(path):
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol["policy"] != {
        "free_max": 0.1,
        "occupied_min": 0.9,
        "argmax_threshold": 0.5,
        "detector_score": 0.6,
        "vehicle_categories": ["car", "bus", "truck", "motorcycle"],
        "D_floor": 0.15,
        "D_margin": 0.08,
        "C_floor": 0.0,
        "C_margin": 0.0,
    }:
        raise ValueError("Unrecognized or changed comparison policy")
    for asset in list(protocol["files"].values()) + [protocol["detector_weights"]]:
        if file_hash(Path(asset["path"])) != asset["sha256"]:
            raise ValueError("Frozen comparison input changed")
    digest = sha256(raw).hexdigest()
    if (path.parent / "protocol.sha256").read_text().split()[0] != digest:
        raise ValueError("Protocol changed")
    return protocol, digest


def acpds_frames(protocol):
    files = protocol["files"]
    manifest = json.loads(Path(files["acpds_manifest"]["path"]).read_text())
    annotations = json.loads(Path(files["acpds_annotations"]["path"]).read_text())
    if manifest["split"] != "official test":
        raise ValueError("Not the frozen official test manifest")
    if manifest["annotations_sha256"] != files["acpds_annotations"]["sha256"]:
        raise ValueError("ACPDS annotation identity mismatch")
    seen = set()
    for split in ("train", "valid", "test"):
        names = annotations[split]["file_names"]
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("ACPDS split overlap")
        seen.update(names)
    test = annotations["test"]
    pairs = dict(
        zip(
            test["file_names"],
            zip(test["rois_list"], test["occupancy_list"], strict=True),
            strict=True,
        )
    )
    frames = []
    for original in manifest["frames"]:
        frame = deepcopy(original)
        name = Path(frame["image"]).name
        if Path(frame["image"]).parts != ("images", name) or name not in pairs:
            raise ValueError("Unexpected ACPDS test path")
        rois, labels = pairs[name]
        if frame["rois"] != rois or frame["labels"] != labels:
            raise ValueError("Test ROI/state differs from pinned publisher annotation")
        if len(rois) != len(labels) or not all(type(x) is bool for x in labels):
            raise ValueError("Unpaired test annotations")
        frame.update(
            cohort="ACPDS-test",
            source_path=f"ACPDS/test/{name}",
            states=["occupied" if x else "free" for x in labels],
            root=protocol["cohorts"]["ACPDS-test"]["root"],
        )
        frames.append(frame)
    if {Path(f["image"]).name for f in frames} != set(pairs):
        raise ValueError("Incomplete official test cohort")
    return frames


def cnr_frames(protocol):
    files = protocol["files"]
    manifest = json.loads(Path(files["cnr_manifest"]["path"]).read_text())
    selected = [
        f for f in manifest["frames"] if f["camera_id"] in {"camera1", "camera3"}
    ]
    subset = dict(
        manifest,
        frames=selected,
        sources={
            k: v for k, v in manifest["sources"].items() if k in {"camera1", "camera3"}
        },
    )
    verifier = module_from_file(
        "_pv_final_cnr_verifier", Path(files["cnr_source_verifier"]["path"])
    )
    verifier.verify(subset, Path(protocol["cohorts"]["CNR-Pisa"]["root"]), set())
    frames = []
    for original in selected:
        frame = deepcopy(original)
        if frame["parking_id"] != "pisa-cnr" or frame["role"] != "external_pilot":
            raise ValueError("Wrong external physical parking cohort")
        scale = np.array([frame["width"] - 1, frame["height"] - 1])
        frame.update(
            cohort="CNR-Pisa",
            root=protocol["cohorts"]["CNR-Pisa"]["root"],
            rois=[(np.asarray(s["roi"]) / scale).tolist() for s in frame["slots"]],
            states=[
                {"empty": "free", "occupied": "occupied"}[s["source_label"]]
                for s in frame["slots"]
            ],
        )
        frames.append(frame)
    return frames


def decode_and_check(frame, blocked_hashes, blocked_pixels, seen_hashes, seen_pixels):
    raw = local_path(Path(frame["root"]), frame["image"]).read_bytes()
    if sha256(raw).hexdigest() != frame["sha256"]:
        raise ValueError("Changed test image")
    if frame["sha256"] in blocked_hashes or frame["sha256"] in seen_hashes:
        raise ValueError("Train/val/test leakage or duplicate image")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid test image")
    height, width = bgr.shape[:2]
    if "width" in frame and (width, height) != (frame["width"], frame["height"]):
        raise ValueError("Test image dimensions changed")
    pixel_hash = sha256(bgr.tobytes() + str(bgr.shape).encode()).hexdigest()
    if pixel_hash in blocked_pixels or pixel_hash in seen_pixels:
        raise ValueError("Re-encoded train/val/test duplicate")
    seen_hashes.add(frame["sha256"])
    seen_pixels.add(pixel_hash)
    frame.update(width=width, height=height)
    rois = np.asarray(frame["rois"], dtype=np.float32)
    if (
        rois.shape != (len(frame["states"]), 4, 2)
        or not np.isfinite(rois).all()
        or rois.min() < 0
        or rois.max() > 1
    ):
        raise ValueError("Invalid held-out ROI")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def assignments(rules, rois, width, height, detections):
    engine = Path(__file__).resolve().parents[3] / "apps/engine/src"
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    from parkvision.domain.geometry.point import Point
    from parkvision.domain.geometry.polygon import Polygon
    from parkvision.domain.slot_projection import SlotProjection
    from parkvision.domain.types import CameraId, ParkingSlotId

    projections = [
        SlotProjection(
            CameraId("heldout-evaluation"),
            ParkingSlotId(str(index)),
            Polygon(tuple(Point(float(x), float(y)) for x, y in roi)),
        )
        for index, roi in enumerate(rois)
    ]
    assigned_d = rules.assign_vehicles(projections, width, height, detections)
    polygons = [np.float32(roi) * np.float32((width - 1, height - 1)) for roi in rois]
    assigned_c = {str(index): [] for index in range(len(rois))}
    for index, detection in enumerate(detections):
        if detection["score"] < 0.6 or detection["category"] not in {
            "car",
            "bus",
            "truck",
            "motorcycle",
        }:
            continue
        box = detection["bbox"]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        ranked = sorted(
            (
                (str(i), rules.intersection_area(box, polygon) / box_area)
                for i, polygon in enumerate(polygons)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if (
            ranked
            and ranked[0][1] > 0
            and ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0) > 0
        ):
            assigned_c[ranked[0][0]].append(index)
    return assigned_c, assigned_d


def manual_results(rows, protocol):
    files = protocol["files"]
    queue_path = Path(files["manual_queue"]["path"])
    queue = json.loads(queue_path.read_text())
    manual = json.loads(Path(files["manual_labels"]["path"]).read_text())
    if (
        manual["queue_sha256"] != file_hash(queue_path)
        or queue["source_manifest_sha256"] != files["acpds_manifest"]["sha256"]
    ):
        raise ValueError("Manual subset provenance mismatch")
    if len(queue["observations"]) != 40 or set(manual["records"]) != {
        r["id"] for r in queue["observations"]
    }:
        raise ValueError("Incomplete fixed manual subset")
    by_id = {
        (r["image"], r["roi_index"]): r for r in rows if r["cohort"] == "ACPDS-test"
    }
    selected = []
    for item in queue["observations"]:
        label = manual["records"][item["id"]]
        row = dict(by_id[item["image"], item["roi_index"]])
        row.update(
            author_state=row["source_state"],
            source_state=label["state"],
            visibility=label["visibility"],
            manual_id=item["id"],
        )
        for model in ("baseline", "candidate"):
            for method in ("raw", "D", "C"):
                row[f"{model}_{method}_visible"] = visibility_gate(
                    row[f"{model}_{method}"], row["visibility"]
                )
        selected.append(row)
    return {
        "n": 40,
        "all_records_evidence": methods_summary(selected),
        "by_visibility": {
            v: methods_summary([r for r in selected if r["visibility"] == v])
            for v in ("clear", "partial", "unreadable")
        },
        "shared_clear_only_gate": {
            model: {
                method: summarize(selected, f"{model}_{method}_visible")
                for method in ("raw", "D", "C")
            }
            for model in ("baseline", "candidate")
        },
        "known_author_operator_disagreements": sum(
            r["source_state"] in {"free", "occupied"}
            and r["source_state"] != r["author_state"]
            for r in selected
        ),
        "records": selected,
    }


def run(protocol_path, device="cuda"):
    protocol, protocol_hash = check_protocol(protocol_path)
    output = protocol_path.parent
    if (output / "comparison.json").exists() or (output / "predictions.jsonl").exists():
        raise ValueError("Completed frozen comparison must not be rerun or overwritten")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("RTX/CUDA unavailable in this process")
    engine = Path(__file__).resolve().parents[3] / "apps/engine/src"
    sys.path.insert(0, str(engine))
    files = protocol["files"]
    frames = acpds_frames(protocol) + cnr_frames(protocol)
    for cohort, expected in protocol["cohorts"].items():
        selected = [f for f in frames if f["cohort"] == cohort]
        if (
            len(selected) != expected["frames"]
            or sum(len(f["states"]) for f in selected) != expected["rois"]
        ):
            raise ValueError("Changed evaluation cohort cardinality")
    training = json.loads(Path(files["training_manifest"]["path"]).read_text())
    original_training_path = Path(
        json.loads(Path(files["candidate_reference"]["path"]).read_text())[
            "dataset_manifest"
        ]
    )
    blocked_hashes, blocked_pixels = set(), set()
    for frame in training["frames"]:
        raw = local_path(original_training_path.parent, frame["image"]).read_bytes()
        if sha256(raw).hexdigest() != frame["sha256"]:
            raise ValueError("Candidate train/val image changed")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        blocked_hashes.add(frame["sha256"])
        blocked_pixels.add(
            sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
        )
    caches = {
        cohort: json.loads(Path(files[key]["path"]).read_text())
        for cohort, key in (
            ("ACPDS-test", "acpds_detections"),
            ("CNR-Pisa", "cnr_detections"),
        )
    }
    for cache in caches.values():
        if (
            cache["weights_sha256"] != protocol["detector_weights"]["sha256"]
            or cache["score_threshold"] != 0.6
        ):
            raise ValueError("Wrong frozen detector cache")
    rules = module_from_file("_pv_final_rules", Path(files["rules"]["path"]))
    baseline_module = module_from_file(
        "_pv_final_baseline", Path(files["baseline_code"]["path"])
    )
    baseline = baseline_module.BaselineModels(Path(files["assets"]["path"]), device)
    code_dir = Path(files["candidate_code___init__.py"]["path"]).parent
    module_from_file("_pv_final_candidate", code_dir / "__init__.py", package=True)
    candidate_module = module_from_file(
        "_pv_final_candidate.model", code_dir / "model.py"
    )
    from importlib import import_module

    candidate_data = import_module("_pv_final_candidate.data")
    checkpoint = torch.load(
        files["candidate"]["path"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint["epoch"] != protocol["candidate_selected_epoch"]
        or checkpoint["classes"] != ["free", "occupied"]
        or checkpoint["manifest_sha256"] != files["training_manifest"]["sha256"]
    ):
        raise ValueError("Frozen candidate lineage mismatch")
    candidate = candidate_module.build_model()
    candidate.load_state_dict(checkpoint["state_dict"])
    candidate.eval().to(device)
    write_json(
        output / "execution-lock.json",
        {
            "protocol_sha256": protocol_hash,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": str(torch.__version__),
            "runner_sha256": file_hash(Path(__file__)),
            "metrics_sha256": file_hash(
                Path(__file__).with_name("comparison_metrics.py")
            ),
            "checkpoint_selected_before_test": True,
            "training_or_threshold_tuning_in_this_command": False,
        },
    )
    rows, seen_hashes, seen_pixels = [], set(), set()
    for frame in frames:
        rgb = decode_and_check(
            frame, blocked_hashes, blocked_pixels, seen_hashes, seen_pixels
        )
        detections = caches[frame["cohort"]]["frames"][frame["sha256"]]
        assigned_c, assigned_d = assignments(
            rules, frame["rois"], frame["width"], frame["height"], detections
        )
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
        prepared = (tensor.float() / 255 - mean) / std
        polygons = np.float32(frame["rois"]) * np.float32(
            (frame["width"] - 1, frame["height"] - 1)
        )
        with torch.inference_mode():
            rois = torch.tensor(frame["rois"], dtype=torch.float32, device=device)
            baseline_scores = torch.cat(
                [
                    baseline.slot(prepared, part).softmax(1)[:, 1]
                    for part in rois.split(32)
                ]
            ).tolist()
            arrays = [
                candidate_data.encode_slot(rgb, points.tolist())[0]
                for points in polygons
            ]
            inputs = torch.stack(
                [candidate_module.tensor_input(array) for array in arrays]
            )
            candidate_scores = torch.cat(
                [
                    candidate(part.to(device)).softmax(1)[:, 1].cpu()
                    for part in inputs.split(32)
                ]
            ).tolist()
        for index, target in enumerate(frame["states"]):
            c_ids, d_ids = assigned_c[str(index)], assigned_d[str(index)]
            row = {
                "cohort": frame["cohort"],
                "source_path": frame["source_path"],
                "image": frame["image"],
                "image_sha256": frame["sha256"],
                "roi_index": index,
                "roi_normalized": frame["rois"][index],
                "source_state": target,
                "visibility": "unassessed",
                "C_detector_ids": c_ids,
                "D_detector_ids": d_ids,
                "detector_C": "occupied" if c_ids else "unknown",
                "detector_D": "occupied" if d_ids else "unknown",
            }
            for model, scores in (
                ("baseline", baseline_scores),
                ("candidate", candidate_scores),
            ):
                score = scores[index]
                row[model + "_score"] = score
                row[model + "_argmax"] = "occupied" if score >= 0.5 else "free"
                for method, ids in (("raw", []), ("C", c_ids), ("D", d_ids)):
                    state, reason = rules.classify(score, ids)
                    row[f"{model}_{method}"] = state
                    row[f"{model}_{method}_reason"] = reason
            rows.append(row)
        print(
            json.dumps(
                {
                    "cohort": frame["cohort"],
                    "image": frame["image"],
                    "roi_count": len(frame["states"]),
                }
            ),
            flush=True,
        )
    report = {
        "protocol_sha256": protocol_hash,
        "cohorts": {
            cohort: methods_summary([r for r in rows if r["cohort"] == cohort])
            for cohort in protocol["cohorts"]
        },
        "pooled_diagnostic_only": methods_summary(rows),
        "per_frame": {
            f["cohort"]
            + ":"
            + f["source_path"]: methods_summary(
                [
                    r
                    for r in rows
                    if r["cohort"] == f["cohort"]
                    and r["source_path"] == f["source_path"]
                ]
            )
            for f in frames
        },
        "manual_visibility": manual_results(rows, protocol),
        "n_images": len(frames),
        "n_rois": len(rows),
        "candidate_checkpoint_sha256": files["candidate"]["sha256"],
        "baseline_checkpoint_sha256": files["baseline"]["sha256"],
        "train_val_overlap": 0,
        "exact_test_duplicates": 0,
        "post_test_tuning_performed": False,
    }
    historical = json.loads(
        Path(files["historical_acpds_baseline"]["path"]).read_text()
    )
    previous = {(r["image"], r["roi_index"]): r for r in historical["observations"]}
    acpds = [r for r in rows if r["cohort"] == "ACPDS-test"]
    differences = [
        abs(r["baseline_score"] - previous[r["image"], r["roi_index"]]["p_occupied"])
        for r in acpds
    ]
    report["historical_baseline_consistency"] = {
        "max_abs_score_difference": max(differences),
        "mean_abs_score_difference": sum(differences) / len(differences),
        "score_band_state_changes": sum(
            r["baseline_raw"] != previous[r["image"], r["roi_index"]]["per_slot"]
            for r in acpds
        ),
        "note": (
            "Fresh baseline and candidate share RGB decoding and GPU execution; "
            "historical baseline used CPU and torchvision JPEG decoding"
        ),
    }
    check_protocol(protocol_path)
    write_json(output / "comparison.json", report)
    with (output / "predictions.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                cohort: {
                    model: result[model]["D_fused"]
                    for model in ("baseline", "candidate")
                }
                for cohort, result in report["cohorts"].items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.protocol, args.device)


if __name__ == "__main__":
    main()

"""Freeze a concrete, grouped vehicle-instance task without occupancy pseudo labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from .data import (
    VEHICLE_NAMES,
    file_hash,
    load_annotation,
    mask_array,
    read_rows,
    source_group,
    verify_manifest,
    write_json,
)
from research.occupancy.training_v1.roboflow import NearIndex, dhash


def pick_families(rows, root, limit):
    """One representative per source name; prefer color over grayscale copies."""
    families = defaultdict(list)
    for row in rows:
        families[(row["archive"], row["original_name"])].append(row)
    selected = []
    for family in sorted(families):
        candidates = families[family]
        if len(candidates) > 1:

            def quality(row):
                image = cv2.imread(str(root / row["file"]), cv2.IMREAD_REDUCED_COLOR_8)
                chroma = np.abs(image[:, :, 0].astype(float) - image[:, :, 1]).mean()
                black = (image.max(2) < 8).mean()
                return (-round(chroma, 3), round(black, 3), row["sha256"])

            candidates.sort(key=quality)
        selected.append(candidates[0])
    selected.sort(key=lambda r: file_hash_string("scene-v3:" + r["id"]))
    return selected[:limit]


def file_hash_string(value):
    from hashlib import sha256

    return sha256(value.encode()).hexdigest()


def prepare(audit_root, extended, output, protected=()):
    output.mkdir(exist_ok=False, parents=True)
    audit = json.loads((audit_root / "audit.json").read_text())
    rows = read_rows(audit_root)
    profiles = json.loads((extended / "audit-summary.json").read_text())
    if profiles["original_audit_sha256"] != file_hash(audit_root / "audit.json"):
        raise ValueError("Extended audit does not describe these source files")
    keys = {Path(a["path"]).name: a["sha256"][:16] for a in audit["archives"]}
    garage = keys["Parking.v8i.coco-segmentation.zip"]
    aerial = keys["Parking spaces and tree canopies.v4i.coco-segmentation.zip"]
    road = keys["vehicle occluded.v1i.coco.zip"]
    cache, candidates = {}, []
    protected_hashes, protected_files = set(), []
    for path in protected:
        document = json.loads(path.read_text())
        protected_files.append({"path": str(path.resolve()), "sha256": file_hash(path)})
        for sample in document.get("samples", document.get("frames", [])):
            for key in ("sha256", "image_sha256"):
                if isinstance(sample.get(key), str):
                    protected_hashes.add(sample[key])
    for row in rows:
        if row["archive"] not in {garage, aerial, road}:
            continue
        if (
            row["protected_pucpr"]
            or row["stock_name_hint"]
            or row["sha256"] in protected_hashes
        ):
            continue
        image, classes, annotations = load_annotation(audit_root, row, cache)
        vehicle = [a for a in annotations if classes[a["category_id"]] in VEHICLE_NAMES]
        supervision = "bbox_only" if row["archive"] == road else "mask"
        if not vehicle:
            # Explicit negative images are included only within the mask corpus,
            # which annotates vehicle instances; no missing-car inference from slots.
            if supervision == "bbox_only":
                continue
        if supervision == "mask":
            try:
                for annotation in vehicle:
                    mask_array(annotation, row["height"], row["width"])
            except (ValueError, TypeError, KeyError):
                continue
        group, evidence = source_group(
            row["original_name"], image.get("extra", {}).get("user_tags", [])
        )
        if row["archive"] == garage:
            # One conservative physical-location group: the complete mechanical
            # parking corpus. Export train/val/test are not independent locations.
            group, evidence, split = (
                "mechanical-garage-entire-export",
                "visually_shared_facility_location_unmapped",
                "train",
            )
        elif row["archive"] == aerial:
            if evidence == "unknown":
                continue
            split = "train" if group == "aerial-city:Paderborn" else "val"
        else:
            if evidence == "unknown":
                continue
            split = "val" if group in {"video:IMG_1084", "video:IMG_0009"} else "train"
        candidates.append(
            {
                **row,
                "group": group,
                "group_evidence": evidence,
                "physical_parking_id": None,
                "camera_id": None,
                "session_id": None,
                "split": split,
                "exporter_split": row["split"],
                "supervision": supervision,
                "raw_staging_role": row["staging_role"],
                "staging_role": "development_" + split,
                "protected_source": False,
                "training_eligible": True,
                "occupancy_training_eligible": False,
                "vehicle_annotation_ids": [a["id"] for a in vehicle],
                "source_classes": {str(k): v for k, v in classes.items()},
                "occupancy_label": None,
                "visibility": "unassessed",
            }
        )
    selected = []
    for archive, split, limit in (
        (garage, "train", 600),
        (aerial, "train", 224),
        (aerial, "val", 20),
        (road, "train", 512),
        (road, "val", 80),
    ):
        selected.extend(
            pick_families(
                [
                    r
                    for r in candidates
                    if r["archive"] == archive and r["split"] == split
                ],
                audit_root,
                limit,
            )
        )
    # Pixel dedup after deterministic source-family selection, including split
    # conflicts. Conflicting pixels are excluded rather than assigned to train.
    duplicates = defaultdict(list)
    for row in selected:
        duplicates[row["pixel_sha256"]].append(row)
    samples = []
    for values in duplicates.values():
        if len({r["split"] for r in values}) == 1:
            samples.append(sorted(values, key=lambda r: r["id"])[0])
    # Conservative retrieval guard: if similar pixels cross roles, remove the
    # whole training source group, not just the single suspicious frame.
    near, by_id, near_pairs, remove_groups = NearIndex(), {}, [], set()
    for row in sorted(samples, key=lambda r: r["id"]):
        image = cv2.imread(str(audit_root / row["file"]), cv2.IMREAD_REDUCED_COLOR_4)
        for other, distance in near.query_add(dhash(image), row["id"]):
            previous = by_id[other]
            if previous["split"] != row["split"]:
                near_pairs.append({"a": other, "b": row["id"], "distance": distance})
                remove_groups.add(
                    (previous if previous["split"] == "train" else row)["group"]
                )
        by_id[row["id"]] = row
    samples = [r for r in samples if r["group"] not in remove_groups]
    annotations = {
        r["annotation_file"]: file_hash(audit_root / r["annotation_file"])
        for r in samples
    }
    manifest = {
        "schema_version": 1,
        "task": "vehicle_instance",
        "classes": ["background", "vehicle_instance"],
        "audit_root": str(audit_root.resolve()),
        "audit_sha256": file_hash(audit_root / "audit.json"),
        "extended_audit_sha256": file_hash(extended / "audit-summary.json"),
        "class_mapping": {name: "vehicle_instance" for name in sorted(VEHICLE_NAMES)},
        "mapping_scope": (
            "generic road-transport instance, not occupancy or legal parking "
            "status; person excluded"
        ),
        "annotation_sha256": annotations,
        "samples": sorted(samples, key=lambda r: r["id"]),
        "protected_manifests": protected_files,
        "near_split_candidates": near_pairs,
        "excluded_similar_training_groups": sorted(remove_groups),
        "split_policy": (
            "Whole mechanical facility train; Paderborn train / all Herne val; "
            "complete road video-prefix groups separated."
        ),
        "split_caveat": (
            "Aerial city tags provide geographic grouping. Road video names "
            "are session hints, not verified physical sites/cameras. Overall "
            "site independence is not established."
        ),
        "test_policy": (
            "No new final test. Existing frozen PUCPR/ACPDS/CNR tests and "
            "checkpoints are never training or validation inputs."
        ),
        "sampling": (
            "Deterministic source-name representatives, prefer color/least "
            "black padding, then identity hash; caps fixed before training."
        ),
        "counts": dict(Counter(r["split"] + ":" + r["supervision"] for r in samples)),
        "source_groups": {
            g: dict(Counter(r["split"] for r in samples if r["group"] == g))
            for g in sorted({r["group"] for r in samples})
        },
        "occupancy_labels_created": 0,
        "occlusion_degree_labels_available": False,
    }
    write_json(output / "manifest.json", manifest)
    verify_manifest(output / "manifest.json")
    print(
        json.dumps({"counts": manifest["counts"], "groups": manifest["source_groups"]}),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", default=[])
    args = parser.parse_args()
    prepare(args.audit, args.extended, args.output, args.protected)


if __name__ == "__main__":
    main()

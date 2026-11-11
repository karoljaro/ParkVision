"""Build/review a small local visibility dataset with conservative site splits."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from research.occupancy.training_v1.data import (
    RECIPE,
    encode_slot,
    file_hash,
    local_path,
    write_json,
)

CLASSES = {
    "clear": "The target slot state can be assessed without material obstruction.",
    "partial-but-readable": (
        "Part is obstructed/degraded, but target state remains assessable."
    ),
    "unreadable": "The target state cannot be established from this frame.",
}
UNASSESSED = "unassessed"


def historical_visibility(label):
    """A known occupancy state is required to map legacy partial to readable."""
    visibility, state = label["visibility"], label["state"]
    if visibility == "partial":
        return "partial-but-readable" if state in {"free", "occupied"} else UNASSESSED
    if visibility not in {"clear", "unreadable"}:
        raise ValueError("Unknown historical visibility label")
    if visibility == "clear" and state not in {"free", "occupied"}:
        return UNASSESSED
    return visibility


def pixel_hash(image):
    return sha256(image.tobytes() + str(image.shape).encode()).hexdigest()


def difference_hash(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grid = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    bits = np.packbits((grid[:, 1:] > grid[:, :-1]).ravel())
    return bits.tobytes().hex()


def assign_splits(records, site_splits):
    """Unknown locations and cross-location duplicates are quarantined."""
    for record in records:
        sites = {p["site_id"] for p in record["provenance"]}
        record["split"] = (
            site_splits.get(next(iter(sites)), "quarantine")
            if len(sites) == 1 and None not in sites and not record["label_conflict"]
            else "quarantine"
        )
        if record["split"] not in {"train", "val", "test", "quarantine"}:
            raise ValueError("Invalid site split")
        # Split is a proposal only. This module never approves data for training.
        record["training_eligible"] = False


def quarantine_duplicate_groups(records, near_pairs):
    """Keep connected duplicate families and whole frames out of split overlap."""
    parents = {r["id"]: r["id"] for r in records}

    def root(identity):
        while parents[identity] != identity:
            identity = parents[identity]
        return identity

    def join(a, b):
        parents[root(a)] = root(b)

    frames = {}
    for record in records:
        frame = record["frame_pixel_sha256"]
        previous = frames.setdefault(frame, record["id"])
        join(previous, record["id"])
    for pair in near_pairs:
        join(pair["a"], pair["b"])
    groups = defaultdict(list)
    for record in records:
        groups[root(record["id"])].append(record)
    for group in groups.values():
        if len({r["split"] for r in group}) > 1:
            for record in group:
                record["split"] = "quarantine"


def summarize(records):
    groups = defaultdict(Counter)
    for r in records:
        groups[r["split"]][r["visibility"]] += 1
    return {
        "unique_records": len(records),
        "visibility": dict(Counter(r["visibility"] for r in records)),
        "proposed_splits": {key: dict(value) for key, value in groups.items()},
        "label_conflicts": sum(r["label_conflict"] for r in records),
        "training_ready": False,
        "gaps": [
            "No independent site-disjoint test set with all visibility classes.",
            "Too few independent unreadable examples; "
            "historical diagnostic selection is biased.",
            "ACPDS physical site/camera/day mapping is unavailable; "
            "quarantine required.",
            "Near-duplicate candidates require operator review; "
            "day/slot repetitions are not independent.",
        ],
    }


def build(index_path, output):
    index = json.loads(index_path.read_text())
    if index["schema_version"] != 1:
        raise ValueError("Unknown source index")
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    (output / "arrays").mkdir()
    (output / "frames").mkdir()
    unique, frames = {}, {}
    for source in index["records"]:
        if source["visibility"] not in {*CLASSES, UNASSESSED}:
            raise ValueError("Invalid visibility class")
        if source.get("region_role") != "physical_parking_slot":
            raise ValueError("Nonparking regions (including T-01) are prohibited")
        if "PUCPR" in source["source_path"].upper():
            raise ValueError("Frozen PUCPR content is prohibited")
        for reference in source["references"]:
            if file_hash(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("Changed source annotation/queue/label")
        image_path = Path(source["image_path"])
        if file_hash(image_path) != source["image_sha256"]:
            raise ValueError("Changed source image")
        rgb = cv2.cvtColor(
            cv2.imread(
                str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            ),
            cv2.COLOR_BGR2RGB,
        )
        height, width = rgb.shape[:2]
        points = np.float32(source["roi_normalized"])
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or points.min() < 0
            or points.max() > 1
        ):
            raise ValueError("Invalid ROI geometry")
        points *= np.float32((width - 1, height - 1))
        array, box = encode_slot(rgb, points.tolist())
        identity = pixel_hash(array)
        frame_id = pixel_hash(rgb)
        if frame_id not in frames:
            relative = f"frames/{frame_id}.png"
            cv2.imwrite(str(output / relative), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames[frame_id] = relative
        provenance = {k: v for k, v in source.items() if k != "visibility"}
        if identity in unique:
            record = unique[identity]
            record["provenance"].append(provenance)
            old = record["visibility"]
            new = source["visibility"]
            if old != UNASSESSED and new != UNASSESSED and old != new:
                record["label_conflict"] = True
                record["visibility"] = UNASSESSED
            elif not record["label_conflict"] and old == UNASSESSED:
                record["visibility"] = new
            continue
        crop = f"crops/{identity}.png"
        array_path = f"arrays/{identity}.npy"
        cv2.imwrite(
            str(output / crop), cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
        )
        np.save(output / array_path, array, allow_pickle=False)
        unique[identity] = {
            "id": identity,
            "visibility": source["visibility"],
            "label_conflict": False,
            "crop": crop,
            "array": array_path,
            "crop_sha256": file_hash(output / crop),
            "frame_sha256": file_hash(output / frames[frame_id]),
            "array_sha256": file_hash(output / array_path),
            "frame": frames[frame_id],
            "frame_pixel_sha256": frame_id,
            "crop_pixel_sha256": identity,
            "crop_box": box,
            "roi_normalized": source["roi_normalized"],
            "dhash": difference_hash(array[:, :, :3]),
            "provenance": [provenance],
        }
    records = sorted(unique.values(), key=lambda r: r["id"])
    assign_splits(records, index["site_splits"])
    near = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 6:
                near.append({"a": a["id"], "b": b["id"], "hamming": distance})
    quarantine_duplicate_groups(records, near)
    manifest = {
        "schema_version": 1,
        "kind": "visibility_dataset_preparation",
        "source_index": str(index_path.resolve()),
        "source_index_sha256": file_hash(index_path),
        "classes": CLASSES,
        "unassessed_is_a_class": False,
        "recipe": RECIPE,
        "site_splits": index["site_splits"],
        "split_status": "proposed_only_no_training_authorized_by_this_manifest",
        "records": records,
        "summary": summarize(records),
        "dedup": {
            "input_records": len(index["records"]),
            "unique_records": len(records),
            "exact_duplicates_merged": len(index["records"]) - len(records),
            "near_duplicate_candidates": near,
            "near_duplicate_hamming_max": 6,
        },
    }
    write_json(output / "manifest.json", manifest)
    write_review(output / "manifest.json", output / "review.html")
    return manifest


def apply_labels(manifest_path, labels_path, output):
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text())
    if labels.get("manifest_sha256") != file_hash(manifest_path):
        raise ValueError("Labels belong to a different manifest")
    if not isinstance(labels.get("operator"), str) or not labels["operator"].strip():
        raise ValueError("Missing operator")
    records = {r["id"]: r for r in manifest["records"]}
    for identity, label in labels["records"].items():
        if identity not in records or label["visibility"] not in {*CLASSES, UNASSESSED}:
            raise ValueError("Unknown record or visibility class")
        record = records[identity]
        record.setdefault("label_history", []).append(
            {
                "previous": record["visibility"],
                "operator": labels["operator"],
                "label_file": str(labels_path.resolve()),
                "sha256": file_hash(labels_path),
                "annotation": label,
            }
        )
        record["visibility"] = label["visibility"]
        record["label_conflict"] = False
        record["training_eligible"] = False
    manifest["summary"] = summarize(list(records.values()))
    manifest["parent_manifest_sha256"] = file_hash(manifest_path)
    # Relative asset paths remain valid; write the revision next to the parent.
    if output.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("Write manifest revision alongside its assets")
    write_json(output, manifest)


def write_review(manifest_path, output):
    if output.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("Write review HTML alongside its assets")
    manifest = json.loads(manifest_path.read_text())
    # Escape closing tags in serialized data; no label text is executed as HTML.
    data = json.dumps(
        {"sha256": file_hash(manifest_path), "records": manifest["records"]},
        ensure_ascii=True,
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("review.html").read_text()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(template.replace("__MANIFEST_DATA__", data))


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    sites, frame_splits, seen = {}, {}, set()
    for record in manifest["records"]:
        if record["id"] in seen or record["visibility"] not in {*CLASSES, UNASSESSED}:
            raise ValueError("Duplicate identity or invalid label")
        seen.add(record["id"])
        array_path = local_path(manifest_path.parent, record["array"])
        array = np.load(array_path, allow_pickle=False)
        if (
            file_hash(array_path) != record["array_sha256"]
            or pixel_hash(array) != record["id"]
        ):
            raise ValueError("Changed prepared pixels")
        for key in ("crop", "frame"):
            if (
                file_hash(local_path(manifest_path.parent, record[key]))
                != record[key + "_sha256"]
            ):
                raise ValueError("Changed review image")
        for source in record["provenance"]:
            site = source["site_id"]
            split = record["split"]
            if split != "quarantine":
                if site is None or sites.setdefault(site, split) != split:
                    raise ValueError("Site leakage or unmapped site")
                frame = record["frame_pixel_sha256"]
                if frame_splits.setdefault(frame, split) != split:
                    raise ValueError("Whole frame leakage")
        if record["training_eligible"]:
            raise ValueError("Preparation cannot approve training")
    return summarize(manifest["records"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("build")
    prepare.add_argument("--index", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("--manifest", type=Path, required=True)
    apply.add_argument("--labels", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.index, args.output)["summary"]))
    elif args.command == "review":
        write_review(args.manifest, args.output)
    elif args.command == "apply":
        apply_labels(args.manifest, args.labels, args.output)
    else:
        print(json.dumps(verify(args.manifest)))


if __name__ == "__main__":
    main()

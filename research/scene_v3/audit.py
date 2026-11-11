"""Extend the lossless v1 audit with RLE, prior datasets and semantic decisions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import cv2

from research.occupancy.training_v1.roboflow import NearIndex, dhash
from .data import file_hash, load_annotation, mask_array, read_rows, write_json


def semantic_task(names):
    names = set(names)
    if "parkinglane" in names:
        return "visible_parking_region_polygon_not_paint"
    if "restricted-area" in names:
        return "roadside_region_candidates_not_verified_legal_vacancy"
    if "Vacant" in names:
        return "mixed_vehicle_masks_and_vacant_ground_polygons"
    if "tree" in names:
        return "aerial_canopy_vehicle_and_parking_regions_not_single_slots"
    if "cng" in names:
        return "vehicle_boxes_without_occlusion_degree_labels"
    return "vehicle_instance_masks"


def extended_audit(root, previous, output):
    output.mkdir(exist_ok=False, parents=True)
    original = json.loads((root / "audit.json").read_text())
    rows = read_rows(root)
    archives, cache, profiles = {}, {}, []
    for source in original["archives"]:
        key = source["sha256"][:16]
        archives[key] = source
        selected = [r for r in rows if r["archive"] == key]
        types, classes, invalid, rle_valid, extras = Counter(), Counter(), [], 0, []
        for row in selected:
            image, names, annotations = load_annotation(root, row, cache)
            extras.append(image.get("extra", {}))
            for annotation in annotations:
                cls = names[annotation["category_id"]]
                segmentation = annotation.get("segmentation")
                kind = (
                    "rle"
                    if isinstance(segmentation, dict)
                    else "polygon" if segmentation else "bbox_only"
                )
                types[kind] += 1
                classes[cls] += 1
                if kind == "rle":
                    try:
                        mask_array(annotation, row["height"], row["width"])
                        rle_valid += 1
                    except (ValueError, TypeError, KeyError) as error:
                        invalid.append(
                            {
                                "image": row["id"],
                                "annotation": annotation["id"],
                                "error": str(error),
                            }
                        )
            if row["protected_pucpr"]:
                invalid.append({"image": row["id"], "error": "protected_pucpr"})
        by_name = Counter(r["original_name"] for r in selected)
        profile = {
            "archive": Path(source["path"]).name,
            "sha256": source["sha256"],
            "images": len(selected),
            "unique_bytes": len({r["sha256"] for r in selected}),
            "unique_pixels": len({r["pixel_sha256"] for r in selected}),
            "source_name_families": len(by_name),
            "family_sizes": dict(Counter(by_name.values())),
            "classes": dict(classes),
            "annotation_types": dict(types),
            "valid_rle_masks": rle_valid,
            "invalid": invalid,
            "exporter_splits": source["splits"],
            "semantic_task": semantic_task(classes),
            "export_capture_dates_are_acquisition_times": False,
            "source_geographic_groups_available": "tree" in classes,
            "site_caveat": (
                "city tags are broader groups, not physical slot IDs"
                if "tree" in classes
                else "physical sites/cameras/days unknown"
            ),
        }
        profiles.append(profile)
        write_json(output / f"{key}-profile.json", profile)
    # All original annotation metadata stays unchanged. Type errors raised by the
    # old importer for RLE reflect its missing support, not corrupt source masks.
    old_rows = read_rows(previous) if previous else []
    global_rows = [("new", root, r) for r in rows] + [
        ("previous", previous, r) for r in old_rows
    ]
    raw_groups, pixel_groups, names = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for cohort, _, row in global_rows:
        identity = {
            "cohort": cohort,
            "archive": row["archive"],
            "id": row["id"],
            "split": row["split"],
        }
        raw_groups[row["sha256"]].append(identity)
        pixel_groups[row["pixel_sha256"]].append(identity)
        names[row["original_name"]].append(identity)

    def cross(groups):
        return [
            v
            for v in groups.values()
            if len({(x["cohort"], x["archive"]) for x in v}) > 1
        ]

    write_json(
        output / "cross-dataset-exact.json",
        {
            "bytes": cross(raw_groups),
            "pixels": cross(pixel_groups),
            "source_name_hints": cross(names),
        },
    )

    def fingerprint(entry):
        cohort, data_root, row = entry
        # Reduced JPEG decode is used only for candidate retrieval, not identity.
        image = cv2.imread(str(data_root / row["file"]), cv2.IMREAD_REDUCED_COLOR_4)
        return cohort, row, dhash(image)

    near, identities, pairs, count, cross_count = NearIndex(), {}, [], 0, 0
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for cohort, row, value in pool.map(fingerprint, global_rows):
            key = cohort + ":" + row["id"]
            for other, distance in near.query_add(value, key):
                count += 1
                old = identities[other]
                if (cohort, row["archive"]) != (old["cohort"], old["archive"]):
                    cross_count += 1
                    pairs.append({"a": other, "b": key, "distance": distance})
            identities[key] = {"cohort": cohort, "archive": row["archive"]}
    write_json(
        output / "cross-dataset-similar.json",
        {
            "method": (
                "reduced-decode 256-bit dHash <=6; candidates, not verified "
                "duplicates"
            ),
            "all_pairs": count,
            "cross_dataset_pairs": cross_count,
            "pairs": pairs,
        },
    )
    # Compare every local archive pair by full SHA and record the missing second
    # v8 explicitly rather than comparing unrelated projects as if they were v8.
    inventory = []
    datasets = Path(original["archives"][0]["path"]).parent.parent
    for path in sorted(datasets.rglob("*.zip")):
        inventory.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
        )
    parking_v8 = [
        x
        for x in inventory
        if re.search(r"parking[ ._-]*v8", Path(x["path"]).name, re.I)
    ]
    v8 = {
        "found": parking_v8,
        "compared": len(parking_v8) >= 2,
        "identical": (
            len({x["sha256"] for x in parking_v8}) == 1
            if len(parking_v8) >= 2
            else None
        ),
    }
    write_json(
        output / "archive-inventory.json",
        {"archives": inventory, "parking_v8_comparison": v8},
    )
    result = {
        "schema_version": 1,
        "images": len(rows),
        "unique_new_pixels": len({r["pixel_sha256"] for r in rows}),
        "profiles": profiles,
        "cross_dataset_exact_pixel_groups": len(cross(pixel_groups)),
        "cross_dataset_similar_pairs": cross_count,
        "parking_v8": v8,
        "original_audit_sha256": file_hash(root / "audit.json"),
    }
    write_json(output / "audit-summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "profiles"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extended_audit(args.audit, args.previous, args.output)


if __name__ == "__main__":
    main()

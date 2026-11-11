"""Read-only ZIP audit and lossless, task-aware staging of local COCO exports.

Exporter splits and class names do not establish site independence or occupancy
supervision. Unresolved provenance stays quarantined, including source families
shared with exporter test. No pseudo labels or inferred physical slot IDs.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha1
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import cv2
import numpy as np

from .data import digest, file_hash, local_path, write_json


def checked_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe ZIP member")
    return path


def canonical_name(image):
    name = image.get("extra", {}).get("name", image["file_name"])
    name = re.sub(r"\.rf\.[a-fA-F0-9]+", "", name)
    return re.sub(
        r"(?:_jpg|_png)?\.(?:jpg|jpeg|png)$|_(?:jpg|png)$", "", name, flags=re.I
    )


def source_family(name):
    """A filename hint, NEVER a verified physical parking/camera/session ID."""
    if re.match(r"^\d+-.*_(morning|afternoon|evening|night)_", name):
        return "unverified-view-" + name.split("-", 1)[0]
    if "_mp4-" in name.lower():
        return re.split(r"_mp4-", name, flags=re.I)[0]
    if name.startswith("KakaoTalk_"):
        return "KakaoTalk-unmapped-location"
    if re.match(r"^\d+_[a-f0-9]{10}_", name, re.I):
        return "flickr-like-" + name.split("_", 1)[0]
    return name


def task_for(categories):
    names = {x["name"] for x in categories}
    if {"parking_space", "blocked_space"} <= names:
        return "slot_geometry_only"
    if "ParkingLines" in names:
        return "paint_positive_polygons_only"
    if {"car", "free"} <= names:
        return "vehicle_and_free_region_boxes_only"
    return "unmapped"


def annotation_errors(record, image, categories):
    errors = []
    if record["category_id"] not in categories:
        errors.append("unknown_category")
    bbox = record.get("bbox", [])
    if len(bbox) != 4 or not np.isfinite(bbox).all() or bbox[2] <= 0 or bbox[3] <= 0:
        errors.append("invalid_bbox")
    else:
        x, y, w, h = bbox
        if (
            x < 0
            or y < 0
            or x + w > image["width"] + 0.01
            or y + h > image["height"] + 0.01
        ):
            errors.append("bbox_outside_image")
    segments = record.get("segmentation", [])
    if not isinstance(segments, list):
        errors.append("unsupported_rle")
    else:
        for segment in segments:
            if len(segment) < 6 or len(segment) % 2 or not np.isfinite(segment).all():
                errors.append("invalid_polygon")
                continue
            points = np.array(segment).reshape(-1, 2)
            if (
                (points < 0).any()
                or (points[:, 0] > image["width"]).any()
                or (points[:, 1] > image["height"]).any()
            ):
                errors.append("polygon_outside_image")
    return sorted(set(errors))


def dhash(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (17, 16))
    return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")


class NearIndex:
    """BK-tree of 256-bit hashes; links are visual candidates, not identity proof."""

    def __init__(self):
        self.root = None

    def query_add(self, value, identity, distance=6):
        matches, pending = [], [self.root] if self.root else []
        while pending:
            node = pending.pop()
            delta = (value ^ node[0]).bit_count()
            if delta <= distance:
                matches.extend((other, delta) for other in node[1])
            pending.extend(
                child
                for key, child in node[2].items()
                if delta - distance <= key <= delta + distance
            )
        if self.root is None:
            self.root = [value, [identity], {}]
        else:
            node = self.root
            while True:
                delta = (value ^ node[0]).bit_count()
                if delta == 0:
                    node[1].append(identity)
                    break
                if delta not in node[2]:
                    node[2][delta] = [value, [identity], {}]
                    break
                node = node[2][delta]
        return matches


def cross_split_groups(groups):
    return [rows for rows in groups.values() if len({r["split"] for r in rows}) > 1]


def audit(inputs: Path, output: Path, pinned_tree: Path | None = None):
    cv2.setNumThreads(1)
    archives = sorted(inputs.glob("*.zip"))
    if not archives:
        raise ValueError("No local ZIP files")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "annotations").mkdir()
    (output / "readmes").mkdir()
    tree = {}
    if pinned_tree:
        for entry in json.loads(pinned_tree.read_text())["tree"]:
            if entry["path"].startswith(("PKLot/", "CNRPark-EXT/")):
                tree[entry["sha"]] = entry["path"]
    summary, by_raw, by_pixels, by_name, families = (
        [],
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    near, near_examples, near_count, near_cross_count = NearIndex(), [], 0, 0
    roles = {}
    image_count = 0
    with (output / "images.jsonl").open("x") as manifest:
        for archive in archives:
            archive_sha = file_hash(archive)
            key = archive_sha[:16]
            source = {
                "path": str(archive.resolve()),
                "sha256": archive_sha,
                "bytes": archive.stat().st_size,
                "splits": {},
                "task": None,
                "training_eligible": False,
                "blockers": [
                    "original_data_rights_and_site_ids_unverified",
                    "no_per_physical_slot_occupancy_supervision",
                ],
            }
            with ZipFile(archive) as package:
                members = package.namelist()
                if len(set(members)) != len(members):
                    raise ValueError("Duplicate ZIP member names")
                for name in members:
                    checked_member(name)
                    if name.lower().endswith((".txt", ".md")):
                        (
                            output / "readmes" / f"{key}-{PurePosixPath(name).name}"
                        ).write_bytes(package.read(name))
                annotation_names = sorted(
                    n for n in members if n.endswith("_annotations.coco.json")
                )
                for annotation_name in annotation_names:
                    split = PurePosixPath(annotation_name).parts[0]
                    if split not in {"train", "valid", "test"}:
                        raise ValueError("Unknown exporter split")
                    raw = package.read(annotation_name)
                    (output / "annotations" / f"{key}-{split}.coco.json").write_bytes(
                        raw
                    )
                    data = json.loads(raw)
                    categories = {x["id"]: x["name"] for x in data["categories"]}
                    if len(categories) != len(data["categories"]):
                        raise ValueError("Duplicate category IDs")
                    source["task"] = task_for(data["categories"])
                    images = {x["id"]: x for x in data["images"]}
                    if len(images) != len(data["images"]):
                        raise ValueError("Duplicate image IDs within COCO split")
                    annotations, ids, issues = defaultdict(list), set(), Counter()
                    class_counts = Counter()
                    for record in data["annotations"]:
                        if record["id"] in ids or record["image_id"] not in images:
                            raise ValueError("Duplicate/orphan annotation")
                        ids.add(record["id"])
                        annotations[record["image_id"]].append(record)
                        class_counts[categories[record["category_id"]]] += 1
                        issues.update(
                            annotation_errors(
                                record, images[record["image_id"]], categories
                            )
                        )
                    source["splits"][split] = {
                        "images": len(images),
                        "annotations": len(ids),
                        "classes": dict(class_counts),
                        "categories": data["categories"],
                        "licenses": data.get("licenses", []),
                        "issues": dict(issues),
                        "annotation_sha256": digest(raw),
                    }
                    for image_id, image in images.items():
                        image_count += 1
                        filename = checked_member(image["file_name"])
                        if len(filename.parts) != 1:
                            raise ValueError("Unexpected nested COCO filename")
                        member = f"{split}/{filename}"
                        raw_image = package.read(member)  # zipfile checks CRC
                        pixels = cv2.imdecode(
                            np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR
                        )
                        if pixels is None or pixels.shape[:2] != (
                            image["height"],
                            image["width"],
                        ):
                            raise ValueError("Image dimensions/decoder mismatch")
                        raw_sha = digest(raw_image)
                        pixel_sha = digest(
                            pixels.tobytes() + str(pixels.shape).encode()
                        )
                        image_path = (
                            output / "images" / (raw_sha + filename.suffix.lower())
                        )
                        if not image_path.exists():
                            image_path.write_bytes(raw_image)
                        identity = f"{key}:{split}:{image_id}"
                        roles[identity] = split
                        name = canonical_name(image)
                        family = source_family(name)
                        reference = {
                            "id": identity,
                            "archive": key,
                            "split": split,
                            "member": member,
                        }
                        by_raw[raw_sha].append(reference)
                        by_pixels[pixel_sha].append(reference)
                        by_name[f"{key}:{name}"].append(reference)
                        families[f"{key}:{family}"].append(reference)
                        matches = near.query_add(dhash(pixels), identity)
                        near_count += len(matches)
                        near_cross_count += sum(
                            roles[other] != split for other, _ in matches
                        )
                        if matches and len(near_examples) < 500:
                            near_examples.append(
                                {"id": identity, "candidates": matches[:5]}
                            )
                        git_blob = sha1(
                            f"blob {len(raw_image)}\0".encode() + raw_image
                        ).hexdigest()
                        original_match = tree.get(git_blob)
                        row = dict(
                            reference,
                            file=str(image_path.relative_to(output)),
                            sha256=raw_sha,
                            pixel_sha256=pixel_sha,
                            width=image["width"],
                            height=image["height"],
                            original_name=name,
                            source_family_hint=family,
                            original_repository_match=original_match,
                            stock_name_hint="istock" in name.lower()
                            or "stock-video" in name.lower(),
                            physical_parking_id=None,
                            camera_id=None,
                            session_id=None,
                            task=source["task"],
                            occupancy_label=None,
                            training_eligible=False,
                            staging_role="heldout" if split == "test" else "quarantine",
                            annotation_file=f"annotations/{key}-{split}.coco.json",
                            source_image_id=image_id,
                            annotation_ids=[a["id"] for a in annotations[image_id]],
                            protected_pucpr=bool(
                                original_match and "/PUCPR/" in original_match
                            ),
                        )
                        manifest.write(json.dumps(row, allow_nan=False) + "\n")
                        if image_count % 200 == 0:
                            print(
                                json.dumps(
                                    {
                                        "images_staged": image_count,
                                        "archive": archive.name,
                                    }
                                ),
                                flush=True,
                            )
            if file_hash(archive) != archive_sha:
                raise ValueError("Source ZIP changed during read")
            summary.append(source)
    duplicates = {
        "exact_bytes": [x for x in by_raw.values() if len(x) > 1],
        "decoded_pixels": [x for x in by_pixels.values() if len(x) > 1],
        "source_name": [x for x in by_name.values() if len(x) > 1],
        "exact_bytes_cross_split": cross_split_groups(by_raw),
        "decoded_pixels_cross_split": cross_split_groups(by_pixels),
        "source_name_cross_split": cross_split_groups(by_name),
        "family_hints_cross_split": cross_split_groups(families),
        "near_pairs": near_count,
        "near_cross_split_pairs": near_cross_count,
        "near_examples": near_examples,
        "near_caveat": (
            "256-bit dHash distance<=6 is a candidate; "
            "similar surfaces are not identity proof"
        ),
    }
    write_json(output / "duplicates.json", duplicates)
    result = {
        "schema_version": 1,
        "archives": summary,
        "images": image_count,
        "unique_raw_images": len(by_raw),
        "unique_decoded_images": len(by_pixels),
        "cross_split_exact_groups": len(duplicates["exact_bytes_cross_split"]),
        "cross_split_pixel_groups": len(duplicates["decoded_pixels_cross_split"]),
        "cross_split_source_name_groups": len(duplicates["source_name_cross_split"]),
        "cross_split_family_hints": len(duplicates["family_hints_cross_split"]),
        "near_pairs": near_count,
        "near_cross_split_pairs": near_cross_count,
        "manifest_sha256": file_hash(output / "images.jsonl"),
        "eligible_occupancy_training_images": 0,
    }
    write_json(output / "audit.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def verify_prepared(output: Path):
    result = json.loads((output / "audit.json").read_text())
    if file_hash(output / "images.jsonl") != result["manifest_sha256"]:
        raise ValueError("Staged manifest changed")
    for source in result["archives"]:
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("Original ZIP changed")
        for split, stats in source["splits"].items():
            filename = f"{source['sha256'][:16]}-{split}.coco.json"
            if (
                file_hash(output / "annotations" / filename)
                != stats["annotation_sha256"]
            ):
                raise ValueError("COCO annotations changed")
    count = 0
    for line in (output / "images.jsonl").open():
        row = json.loads(line)
        if row["training_eligible"] or row["occupancy_label"] is not None:
            raise ValueError("Unreviewed export promoted into occupancy training")
        if file_hash(local_path(output, row["file"])) != row["sha256"]:
            raise ValueError("Staged image changed")
        count += 1
    if count != result["images"]:
        raise ValueError("Wrong number of staged images")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pinned-tree", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_prepared(args.output)
        print("Original archives and staged COCO verified")
    else:
        if args.input is None:
            parser.error("--input required for staging")
        audit(args.input, args.output, args.pinned_tree)


if __name__ == "__main__":
    main()

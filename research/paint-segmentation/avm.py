"""Import publisher AVM SS labels into a separate, binary research dataset.

Never modifies the archive or Dataset A. Original random splits are metadata,
not independent validation. Exact RGB matching; uncertain colors stay ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from prepare import read_json, sha256, write_image, write_json

CLASSES = {"background": 0, "paint": 1, "ignore": 255}
RGB_CLASSES = {
    (0, 0, 255): (0, "free_space"),
    (255, 0, 0): (0, "vehicle"),
    (255, 255, 255): (1, "marker"),
    (0, 255, 0): (255, "other"),
    (0, 0, 0): (255, "ego"),
}
SCHEMA = "parkvision-external-avm-v1"


def map_mask(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("AVM source mask must be uint8 RGB, three channels")
    target = np.full(rgb.shape[:2], 255, np.uint8)
    for color, (value, _) in RGB_CLASSES.items():
        target[np.all(rgb == color, axis=2)] = value
    return target


def decode(data: bytes) -> np.ndarray:
    value = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if value is None or value.dtype != np.uint8 or value.shape != (160, 320, 3):
        raise ValueError("Expected native 320x160 uint8 three-channel image/mask")
    return value


def import_archive(archive: Path, output: Path, provenance: Path, count=6763) -> dict:
    if output.exists():
        raise ValueError("Import output must be a NEW directory")
    source = read_json(provenance)
    archive_hash = sha256(archive)
    if source["archive_sha256"] != archive_hash:
        raise ValueError("Source ZIP hash differs from provenance")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(set(names)) != len(names) or any(
            PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts
            for n in names
        ):
            raise ValueError("Unsafe or duplicate ZIP member")
        families = {}
        for directory, extension in (
            ("images", "jpg"),
            ("gt", "png"),
            ("annotations", "yml"),
        ):
            families[directory] = {
                Path(n).stem: n
                for n in names
                if re.fullmatch(rf"{directory}/\d{{8}}\.{extension}", n)
            }
        ids = set(families["images"])
        if len(ids) != count or any(set(f) != ids for f in families.values()):
            raise ValueError("Image/mask/annotation IDs must match expected count")
        conditions = {}
        for line in zipped.read("info.txt").decode().splitlines():
            image_member, *codes = line.split()
            if len(codes) != 3 or Path(image_member).stem in conditions:
                raise ValueError("Malformed or duplicate conditions record")
            conditions[Path(image_member).stem] = [int(c) for c in codes]
        original_splits = {}
        for split in ("train", "test"):
            for line in zipped.read(f"{split}_db.txt").decode().splitlines():
                image_member, mask_member = line.split()
                sample_id = Path(image_member).stem
                if sample_id in original_splits or sample_id != Path(mask_member).stem:
                    raise ValueError("Malformed or overlapping original split")
                original_splits[sample_id] = split
        if set(conditions) != ids or set(original_splits) != ids:
            raise ValueError("Metadata must cover all paired images")
        output.mkdir(parents=True)
        for directory in ("images", "source-masks", "source-annotations", "masks"):
            (output / directory).mkdir()
        files, samples = [], []
        palette: Counter = Counter()
        unknown_frames: Counter = Counter()
        counts: Counter = Counter()
        decoded_groups: dict[str, list[str]] = defaultdict(list)
        previous = None
        previous_id = None
        near_pairs = []
        for number, sample_id in enumerate(sorted(ids), 1):
            record: dict[str, Any] = {
                "id": sample_id,
                "width": 320,
                "height": 160,
                "role": "external_train",
                "parking_id": "unknown",
                "camera_id": "unknown",
                "session_id": "unknown",
                "group_id": "avm-all-unknown-scenes",
                "original_random_split": original_splits[sample_id],
                "condition_codes": conditions[sample_id],
            }
            decoded = {}
            for family, key, directory in (
                ("images", "image", "images"),
                ("gt", "source_mask", "source-masks"),
                ("annotations", "source_annotation", "source-annotations"),
            ):
                member = families[family][sample_id]
                data = zipped.read(member)  # CRC verified by ZipFile.read
                path = output / directory / Path(member).name
                path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if sha256(path) != digest:
                    raise ValueError("Original file copy differs")
                record[key] = str(path.relative_to(output))
                record[f"{key}_sha256"] = digest
                files.append({"path": record[key], "sha256": digest})
                if family != "annotations":
                    decoded[key] = decode(data)
            image = decoded["image"]
            rgb = cv2.cvtColor(decoded["source_mask"], cv2.COLOR_BGR2RGB)
            colors, frequencies = np.unique(
                rgb.reshape(-1, 3), axis=0, return_counts=True
            )
            for color, frequency in zip(colors, frequencies):
                color_key = tuple(int(c) for c in color)
                palette[color_key] += int(frequency)
                if color_key not in RGB_CLASSES:
                    unknown_frames[color_key] += 1
            mask = map_mask(rgb)
            record["mask"] = f"masks/{sample_id}.png"
            path = output / record["mask"]
            write_image(path, mask)
            reread = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                reread is None
                or not np.array_equal(mask, reread)
                or reread.dtype != np.uint8
            ):
                raise ValueError("Indexed mask round-trip differs")
            record["mask_sha256"] = sha256(path)
            files.append({"path": record["mask"], "sha256": record["mask_sha256"]})
            record["pixel_counts"] = {
                name: int(np.count_nonzero(mask == value))
                for name, value in CLASSES.items()
            }
            counts.update(record["pixel_counts"])
            decoded_groups[hashlib.sha256(image.tobytes()).hexdigest()].append(
                sample_id
            )
            if previous is not None and previous_id is not None:
                mad = float(
                    np.abs(image.astype(np.int16) - previous.astype(np.int16)).mean()
                )
                if mad < 3:
                    near_pairs.append(
                        {
                            "ids": [previous_id, sample_id],
                            "rgb_mad": mad,
                            "original_split_crossing": original_splits[previous_id]
                            != original_splits[sample_id],
                        }
                    )
            previous, previous_id = image, sample_id
            samples.append(record)
            if number % 500 == 0:
                print(f"validated {number}/{count}", flush=True)
        for name in (
            "info.txt",
            "train_db.txt",
            "test_db.txt",
            "split_db.py",
            "show_db_info.py",
        ):
            # Preserve publisher metadata/scripts as bytes; never execute them.
            data = zipped.read(name)
            (output / name).write_bytes(data)
            files.append({"path": name, "sha256": sha256(output / name)})
    report = {
        "images": len(samples),
        "native_dimensions": [320, 160],
        "pixel_counts": dict(counts),
        "curb_pixels": 0,
        "source_rgb_counts": {str(k): v for k, v in sorted(palette.items())},
        "unknown_color_frame_counts": {str(k): v for k, v in unknown_frames.items()},
        "exact_decoded_duplicates": [v for v in decoded_groups.values() if len(v) > 1],
        "adjacent_near_duplicates_mad_lt_3": near_pairs,
        "conditions_counts": dict(Counter(str(s["condition_codes"]) for s in samples)),
        "leakage_policy": (
            "All AVM is external training; no AVM validation or test claim. "
            "IDs unknown. Adjacency MAD is a diagnostic, not proof of independence."
        ),
    }
    dataset = {
        "schema": SCHEMA,
        "classes": CLASSES,
        "provenance": source,
        "samples": samples,
        "report": report,
    }
    write_json(output / "dataset.json", dataset)
    write_json(output / "validation.json", report)
    write_json(output / "provenance.json", source)
    for name in ("dataset.json", "validation.json", "provenance.json"):
        files.append({"path": name, "sha256": sha256(output / name)})
    write_json(
        output / "lock.json",
        {"schema": SCHEMA, "files": files, "archive_sha256": archive_hash},
    )
    verify_dataset(output)
    return report


def verify_dataset(root: Path) -> dict:
    lock = read_json(root / "lock.json")
    if lock.get("schema") != SCHEMA:
        raise ValueError("Unsupported AVM lock")
    for item in lock["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen AVM file changed: {item['path']}")
    dataset = read_json(root / "dataset.json")
    if dataset.get("schema") != SCHEMA or dataset.get("classes") != CLASSES:
        raise ValueError("AVM schema/classes changed")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    print(import_archive(args.archive, args.output, args.provenance))


if __name__ == "__main__":
    main()

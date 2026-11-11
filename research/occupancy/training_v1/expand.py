"""Expand a frozen UFPR training set with official ACPDS train, preserving val.

The publisher's parking-disjoint official split is retained wholesale. Exact
site IDs are unavailable and are not invented. No test images are opened.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from .acpds import add_official_split
from .data import file_hash, local_path, verify, write_json
from .roboflow import dhash


def link_verified(source: Path, destination: Path, expected: str):
    if file_hash(source) != expected:
        raise ValueError("Base dataset file changed")
    if destination.exists():
        if file_hash(destination) != expected:
            raise ValueError("Existing expanded dataset file changed")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def clone_base(base: Path, output: Path, manifest: dict):
    """Immutable byte-identical references; no writes to inherited input files."""
    entries = {}
    for frame in manifest["frames"]:
        entries[frame["image"]] = frame["sha256"]
    for sample in manifest["samples"]:
        entries[sample["array"]] = sample["sha256"]
    for source in manifest["sources"].values():
        entries[source["archive"]] = source["sha256"]
    for relative, expected in entries.items():
        link_verified(
            local_path(base, relative), local_path(output, relative), expected
        )


def inspect_images(output, manifest):
    """Cross-role near-duplicate candidates and orientation audit, not tuning."""
    from PIL import Image

    fingerprints = {"train": [], "val": []}
    orientation_counts = Counter()
    for frame in manifest["frames"]:
        path = local_path(output, frame["image"])
        with Image.open(path) as image:
            orientation = image.getexif().get(274, 1)
        orientation_counts[str(orientation)] += 1
        if orientation != 1:
            raise ValueError(
                "Nontrivial EXIF orientation needs an explicit ROI transform"
            )
        raw = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        fingerprints[frame["split"]].append((frame["source_path"], dhash(image)))
    candidates = [
        {"train": a, "val": b, "hamming_256": (x ^ y).bit_count()}
        for a, x in fingerprints["train"]
        for b, y in fingerprints["val"]
        if (x ^ y).bit_count() <= 6
    ]
    return {
        "exif_orientations": dict(orientation_counts),
        "cross_split_near_candidates": candidates,
        "near_rule": "256-bit grayscale dHash Hamming <=6; requires visual review",
    }


def expand(workspace: Path, base: Path, output: Path):
    base_manifest_path = base / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    verify(base, base_manifest)
    if any(s["camera_id"] == "ACPDS-train" for s in base_manifest["samples"]):
        raise ValueError("Base already contains ACPDS train")
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["expansion"]["base_manifest_sha256"] != file_hash(
            base_manifest_path
        ):
            raise ValueError("Expanded dataset belongs to another base")
        verify(output, manifest)
        print(json.dumps(manifest["counts"]), flush=True)
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    clone_base(base, output, base_manifest)
    manifest = deepcopy(base_manifest)
    manifest["expansion"] = {
        "base_manifest": str(base_manifest_path.resolve()),
        "base_manifest_sha256": file_hash(base_manifest_path),
        "added_source": "ACPDS official train only",
        "license": "MIT, explicitly including dataset in Marek 2021",
        "publisher_protocol": "https://arxiv.org/abs/2107.12207",
        "physical_site_separation": (
            "publisher-declared parking-disjoint official splits"
        ),
        "physical_site_ids": None,
        "source_sha256": file_hash(Path(__file__)),
    }
    add_official_split(workspace, output, manifest, "train", cache_roots=(base,))
    # The original val rows, images and arrays must stay bit-for-bit identical.
    for collection in ("frames", "samples"):
        before = [x for x in base_manifest[collection] if x["split"] == "val"]
        after = [x for x in manifest[collection] if x["split"] == "val"]
        if before != after:
            raise ValueError("Validation was changed during expansion")
    manifest["counts"] = {
        "frames": dict(Counter(f["split"] for f in manifest["frames"])),
        "samples": dict(Counter(s["split"] for s in manifest["samples"])),
        "classes": {
            split: dict(
                Counter(
                    str(s["label"]) for s in manifest["samples"] if s["split"] == split
                )
            )
            for split in ("train", "val")
        },
        "train_sources": dict(
            Counter(
                s["camera_id"] for s in manifest["samples"] if s["split"] == "train"
            )
        ),
        "acquisition": dict(
            Counter(f.get("acquisition", "inherited_base") for f in manifest["frames"])
        ),
    }
    inspection = inspect_images(output, manifest)
    if inspection["cross_split_near_candidates"]:
        write_json(output / "near-duplicates-review.json", inspection)
        raise ValueError("Cross-split near-duplicate candidates require review")
    verify(output, manifest)
    (output / "source/expand.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "inspection.json", inspection)
    write_json(output / "manifest.partial.json", manifest)
    (output / "manifest.partial.json").replace(output / "manifest.json")
    print(json.dumps(manifest["counts"]), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expand(args.workspace, args.base_data, args.output)


if __name__ == "__main__":
    main()

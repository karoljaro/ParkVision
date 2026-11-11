"""COCO mask semantics and conservative source grouping for scene research."""

import json
from pathlib import Path
import re

import numpy as np
from pycocotools import mask as coco_mask

from research.occupancy.training_v1.data import file_hash, local_path, write_json

VEHICLE_NAMES = {
    "car",
    "vehicle",
    "drivingvehicle",
    "truck",
    "trailer",
    "bus",
    "bicycle",
    "motorcycle",
    "cng",
    "rickshaw",
    "other-vehicle",
}


def mask_array(annotation, height, width):
    """Decode genuine polygons/RLE. Bbox-only data never becomes a fake mask."""
    segmentation = annotation.get("segmentation")
    if not segmentation:
        raise ValueError("No instance mask; a bounding box is not a silhouette")
    if isinstance(segmentation, list):
        for points in segmentation:
            if len(points) < 6 or len(points) % 2 or not np.isfinite(points).all():
                raise ValueError("Invalid polygon")
        encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        if segmentation.get("size") != [height, width]:
            raise ValueError("RLE dimensions differ from image")
        if isinstance(segmentation.get("counts"), list):
            encoded = coco_mask.frPyObjects(segmentation, height, width)
        else:
            encoded = dict(segmentation)
            if isinstance(encoded.get("counts"), str):
                encoded["counts"] = encoded["counts"].encode("ascii")
    else:
        raise ValueError("Unsupported segmentation representation")
    result = coco_mask.decode(encoded)
    if result.shape != (height, width) or not result.any():
        raise ValueError("Empty or invalid decoded mask")
    return result.astype(np.uint8)


def source_group(name, tags=()):
    """Conservative groups; video filenames are hints, never verified site IDs."""
    if name.startswith("dop10rgbi_"):
        cities = sorted(set(tags) & {"Paderborn", "Herne"})
        if len(cities) != 1:
            return "aerial-unmapped", "unknown"
        # Whole city, all parent tiles/rotations stay in one group.
        return "aerial-city:" + cities[0], "city_tag_and_grid_filename"
    if name.startswith("IMG_") and re.search(r"-\d+$", name):
        return "video:" + re.sub(r"-\d+$", "", name), "filename_session_hint"
    return "unknown-source", "unknown"


def read_rows(audit_root):
    return [json.loads(line) for line in (audit_root / "images.jsonl").open()]


def load_annotation(audit_root, row, cache):
    key = row["annotation_file"]
    if key not in cache:
        raw = json.loads(local_path(audit_root, key).read_text())
        categories = {c["id"]: c["name"] for c in raw["categories"]}
        annotations = {}
        for annotation in raw["annotations"]:
            annotations.setdefault(annotation["image_id"], []).append(annotation)
        cache[key] = (raw, categories, annotations)
    raw, categories, annotations = cache[key]
    image = next(x for x in raw["images"] if x["id"] == row["source_image_id"])
    return image, categories, annotations.get(row["source_image_id"], [])


def verify_manifest(path):
    data = json.loads(path.read_text())
    if data["schema_version"] != 1 or data["task"] != "vehicle_instance":
        raise ValueError("Wrong scene dataset contract")
    owners, pixels, ids = {}, {}, set()
    root = Path(data["audit_root"])
    protected_hashes = set()
    for reference in data.get("protected_manifests", []):
        source = Path(reference["path"])
        if file_hash(source) != reference["sha256"]:
            raise ValueError("Protected manifest changed")
        document = json.loads(source.read_text())
        for sample in document.get("samples", document.get("frames", [])):
            protected_hashes.update(
                sample[key]
                for key in ("sha256", "image_sha256")
                if isinstance(sample.get(key), str)
            )
    for row in data["samples"]:
        if (
            row["id"] in ids
            or row["protected_source"]
            or row["sha256"] in protected_hashes
        ):
            raise ValueError("Duplicate or protected source")
        ids.add(row["id"])
        for key in (row["group"], row["pixel_sha256"]):
            if owners.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Group or pixel leakage")
        if row["pixel_sha256"] in pixels:
            raise ValueError("Duplicate training pixels")
        pixels[row["pixel_sha256"]] = row["id"]
        if file_hash(local_path(root, row["file"])) != row["sha256"]:
            raise ValueError("Changed staged image")
    for name, expected in data["annotation_sha256"].items():
        if file_hash(local_path(root, name)) != expected:
            raise ValueError("Changed source annotations")
    return data


__all__ = ["file_hash", "local_path", "write_json"]

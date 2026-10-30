"""Standalone LabelMe polygon -> indexed PNG workflow; never imports the engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
CLASSES = {"background": 0, "paint": 1, "curb": 2, "ignore": 255}
# OpenCV BGR: background blue, paint magenta, curb orange, ignore unchanged.
COLORS = {0: (180, 80, 0), 1: (255, 0, 255), 2: (0, 165, 255)}
SPLITS = {"pilot", "train", "val", "test"}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def read_image(path: Path) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cast(Image, image)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")


def write_image(path: Path, image: Image) -> None:
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Cannot write image: {path}")


def validate_mask(mask: Image, shape: tuple[int, int]) -> None:
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != shape:
        raise ValueError("Mask must be uint8 HxW at the original image dimensions")
    if not set(np.unique(mask).tolist()) <= set(CLASSES.values()):
        raise ValueError("Mask has unknown class IDs (allowed: 0, 1, 2, 255)")
    if not np.any(mask != 255):
        raise ValueError("Mask contains no supervised pixels")


def polygon_mask(shape: dict[str, Any], height: int, width: int) -> Image:
    if shape.get("shape_type") != "polygon":
        raise ValueError("Only polygon shapes are accepted; no points/bboxes/lines")
    if shape.get("mask") is not None:
        raise ValueError("Embedded shape masks are unsupported; use polygons")
    try:
        points = np.asarray(shape["points"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid polygon points") from error
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Polygon needs at least three x,y points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Non-finite polygon coordinate")
    if (
        np.any(points < 0)
        or np.any(points[:, 0] > width - 1)
        or np.any(points[:, 1] > height - 1)
    ):
        raise ValueError("Polygon outside image pixel centres; no silent clipping")
    # Nearest pixel centre, ties to even; filled edges included, no antialiasing.
    vertices = np.rint(points).astype(np.int32)
    if abs(cv2.contourArea(vertices)) == 0:
        raise ValueError("Degenerate polygon after rounding to pixel centres")
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(result, [vertices], 1, lineType=cv2.LINE_8)
    return result


def labelme_mask(annotation: dict[str, Any], image: Image) -> Image:
    height, width = image.shape[:2]
    if (annotation.get("imageHeight"), annotation.get("imageWidth")) != (
        height,
        width,
    ):
        raise ValueError("LabelMe image dimensions differ from the decoded image")
    if annotation.get("imageData") is not None:
        raise ValueError("Use LabelMe --nodata; embedded imageData may be stale")
    flags = annotation.get("flags")
    if not isinstance(flags, dict) or flags.get("reviewed") is not True:
        raise ValueError("Set the image-level reviewed flag after manual review")
    shapes = annotation.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("LabelMe shapes must be a list")
    regions = {name: np.zeros((height, width), dtype=np.uint8) for name in CLASSES}
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("label") not in CLASSES:
            raise ValueError("Unknown shape label; use background/paint/curb/ignore")
        regions[shape["label"]] |= polygon_mask(shape, height, width)
    conflict = regions["paint"] & regions["curb"] & (1 - regions["ignore"])
    if np.any(conflict):
        raise ValueError("Paint and curb overlap; correct them or mark ignore")
    mask = np.full((height, width), 255, dtype=np.uint8)
    # Order-independent precedence, unlike LabelMe's ordinary painter order.
    for name, value in CLASSES.items():
        mask[regions[name] != 0] = value
    validate_mask(mask, (height, width))
    return mask


def overlay(image: Image, mask: Image) -> Image:
    validate_mask(mask, image.shape[:2])
    result = image.copy()
    for value, color in COLORS.items():
        selected = mask == value
        result[selected] = np.rint(
            image[selected].astype(np.float64) * 0.5
            + np.asarray(color, dtype=np.float64) * 0.5
        ).astype(np.uint8)
    return result


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    if manifest.get("version") != 1:
        raise ValueError("Expected manifest version 1")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest needs a nonempty samples list")
    ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be an object")
        for key in (
            "id",
            "image",
            "annotation",
            "parking_id",
            "camera_id",
            "group_id",
            "split",
            "surface",
            "source",
            "license",
            "image_sha256",
        ):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"Missing/non-string sample field: {key}")
        sample_id = sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id) or sample_id in ids:
            raise ValueError(f"Unsafe or duplicate sample id: {sample_id}")
        ids.add(sample_id)
        if not re.fullmatch(r"[a-f0-9]{64}", sample["image_sha256"]):
            raise ValueError("image_sha256 must contain 64 lowercase hex characters")
        if sample["split"] not in SPLITS:
            raise ValueError("split must be pilot/train/val/test")
        if sample["surface"] not in {"wet", "dry", "unknown"}:
            raise ValueError("surface must be wet/dry/unknown")
        conditions = sample.get("conditions")
        if not isinstance(conditions, list) or not all(
            isinstance(value, str) for value in conditions
        ):
            raise ValueError("conditions must be a list of strings")
        if sample["split"] != "pilot" and any(
            sample[key] == "unknown"
            for key in ("parking_id", "camera_id", "group_id", "source", "license")
        ):
            raise ValueError("Non-pilot samples need verified identity/source/license")
    return samples


def check_leakage(samples: list[dict[str, Any]], root: Path) -> None:
    """Conservative site/camera/related-frame grouping plus exact decoded duplicates."""
    ownership: dict[tuple[str, str], str] = {}
    for sample in samples:
        image_path = root / sample["image"]
        if sha256(image_path) != sample["image_sha256"]:
            raise ValueError(f"{sample['id']}: image_sha256 mismatch; image changed")
        image = read_image(image_path)
        pixels = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        keys = [("pixels", pixels), ("path", str((root / sample["image"]).resolve()))]
        for field in ("parking_id", "camera_id", "group_id"):
            if sample[field] != "unknown":
                keys.append((field, sample[field]))
        for key in keys:
            previous = ownership.setdefault(key, sample["split"])
            if previous != sample["split"]:
                raise ValueError(f"Split leakage for {key[0]}: {key[1]}")


def inspect_manifest(path: Path) -> dict[str, Any]:
    samples = load_manifest(path)
    check_leakage(samples, path.parent)
    records = []
    for sample in samples:
        image = read_image(path.parent / sample["image"])
        annotation = path.parent / sample["annotation"]
        records.append(
            {
                "id": sample["id"],
                "width": image.shape[1],
                "height": image.shape[0],
                "split": sample["split"],
                "surface": sample["surface"],
                "annotation_exists": annotation.is_file(),
            }
        )
    return {"samples": records, "note": "Inventory only; not segmentation metrics"}


def convert(path: Path, output: Path) -> dict[str, Any]:
    samples = load_manifest(path)
    check_leakage(samples, path.parent)
    # Validate the whole batch before creating outputs; never overwrite a dataset.
    for sample in samples:
        image_path = (path.parent / sample["image"]).resolve()
        annotation_path = (path.parent / sample["annotation"]).resolve()
        annotation = read_json(annotation_path)
        if (annotation_path.parent / annotation.get("imagePath", "")).resolve() != (
            image_path
        ):
            raise ValueError(
                f"{sample['id']}: LabelMe imagePath references another image"
            )
        labelme_mask(annotation, read_image(image_path))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for sample in samples:
        image_path = (path.parent / sample["image"]).resolve()
        annotation_path = (path.parent / sample["annotation"]).resolve()
        image = read_image(image_path)
        mask = labelme_mask(read_json(annotation_path), image)
        mask_path = output / f"{sample['id']}.mask.png"
        preview_path = output / f"{sample['id']}.overlay.png"
        write_image(mask_path, mask)
        # Verify lossless serialization rather than assuming an encoder round trip.
        decoded = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if decoded is None or not np.array_equal(decoded, mask):
            raise ValueError(f"PNG round trip changed pixels: {mask_path}")
        write_image(preview_path, overlay(image, mask))
        records.append(
            {
                **sample,
                "image": str(image_path),
                "annotation": str(annotation_path),
                "mask": mask_path.name,
                "overlay": preview_path.name,
                "image_sha256": sha256(image_path),
                "annotation_sha256": sha256(annotation_path),
                "mask_sha256": sha256(mask_path),
                "width": image.shape[1],
                "height": image.shape[0],
                "pixels": {
                    name: int(np.count_nonzero(mask == value))
                    for name, value in CLASSES.items()
                },
            }
        )
    report = {
        "version": 1,
        "classes": CLASSES,
        "samples": records,
        "manifest_sha256": sha256(path),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "note": "Annotation counts, not model quality",
    }
    write_json(output / "dataset.json", report)
    return report


def initialize(images: list[Path], output: Path) -> None:
    if len({image.stem for image in images}) != len(images):
        raise ValueError("Input image stems must be unique")
    for image_path in images:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_path.stem):
            raise ValueError(
                f"Use an alphanumeric/underscore/hyphen stem: {image_path}"
            )
        read_image(image_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    samples = []
    tiles = []
    for image_path in images:
        target = output / "images" / image_path.name
        shutil.copyfile(image_path, target)
        samples.append(
            {
                "id": image_path.stem,
                "image": f"images/{target.name}",
                "annotation": f"images/{target.stem}.json",
                "split": "pilot",
                "parking_id": "unknown",
                "camera_id": "unknown",
                "group_id": "unknown",
                "surface": "unknown",
                "conditions": [],
                "source": str(image_path.resolve()),
                "license": "unknown",
                "image_sha256": sha256(target),
            }
        )
        image = read_image(target)
        height, width = image.shape[:2]
        scale = min(640 / width, 440 / height)
        thumb = cv2.resize(image, (round(width * scale), round(height * scale)))
        tile = np.full((480, 640, 3), 32, dtype=np.uint8)
        tile[: thumb.shape[0], : thumb.shape[1]] = thumb
        cv2.putText(
            tile,
            f"{image_path.stem} ({width}x{height})",
            (8, 468),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    write_json(output / "manifest.json", {"version": 1, "samples": samples})
    write_image(output / "overview.jpg", np.vstack(tiles))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Copy real images for manual annotation")
    init.add_argument("--images", nargs="+", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Inventory images and check splits")
    inspect.add_argument("manifest", type=Path)
    export = commands.add_parser("convert", help="Validate LabelMe and export PNGs")
    export.add_argument("manifest", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.images, args.output)
            print(f"Prepared {len(args.images)} real images; no annotations generated")
        elif args.command == "inspect":
            print(json.dumps(inspect_manifest(args.manifest), indent=2))
        else:
            report = convert(args.manifest, args.output)
            print(
                f"Exported {len(report['samples'])} masks and overlays to {args.output}"
            )
    except (OSError, ValueError, TypeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

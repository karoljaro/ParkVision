"""Reproduce the raster and blank-image supplement to the lossless COCO audit."""

import argparse
from collections import Counter
from pathlib import Path

import cv2

from .data import file_hash, load_annotation, mask_array, read_rows, write_json


def check(root, output):
    rows, cache, issues, blank = read_rows(root), {}, [], []
    checked = Counter()
    for row in rows:
        _, names, annotations = load_annotation(root, row, cache)
        for annotation in annotations:
            if not annotation.get("segmentation"):
                continue
            checked[row["archive"]] += 1
            try:
                mask_array(annotation, row["height"], row["width"])
            except (ValueError, TypeError, KeyError) as error:
                issues.append(
                    {
                        "image": row["id"],
                        "annotation": annotation["id"],
                        "class": names[annotation["category_id"]],
                        "error": str(error),
                    }
                )
        image = cv2.imread(str(root / row["file"]), cv2.IMREAD_REDUCED_COLOR_8)
        if (image.max(2) < 8).mean() > 0.99:
            blank.append(row["id"])
    result = {
        "audit_sha256": file_hash(root / "audit.json"),
        "checked_masks_by_archive": dict(checked),
        "invalid_masks": issues,
        "almost_black_image_ids": blank,
        "blank_criterion": "reduced JPEG decode; >99% pixels have max(B,G,R)<8",
        "policy": (
            "Invalid target supervision is rejected, never converted to background. "
            "Blank pixels are ignored by map loss. No occupancy labels are inferred."
        ),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.audit, args.output)
    print(
        {
            "invalid_masks": len(result["invalid_masks"]),
            "almost_black_images": len(result["almost_black_image_ids"]),
        }
    )


if __name__ == "__main__":
    main()

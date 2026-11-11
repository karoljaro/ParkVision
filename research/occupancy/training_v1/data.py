"""Pinned MetaPKLot physical-slot import, conservative grouping and integrity.

Train uses UFPR04/05 and optionally ACPDS train; val uses official ACPDS valid.
PUCPR test content is never used for training, validation or tuning.
Source empty/occupied supervises classification, not visibility or legal status.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

import cv2
import numpy as np

COMMIT = "9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2"
BASE = f"https://raw.githubusercontent.com/DSBD-Research/MetaPKLot-Dataset/{COMMIT}/"
VIEWS = {
    "UFPR04": ("PKLot", "ufpr04_spots", "ufpr", "train", "CC-BY-4.0"),
    "UFPR05": ("PKLot", "ufpr05_spots", "ufpr", "train", "CC-BY-4.0"),
}
RECIPE = {"size": 224, "context": 0.25, "format": "RGB+ROI-mask-letterbox-v1"}


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blob_hash(raw: bytes) -> str:
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Path escapes dataset")
    return path


def source_identity(path: str) -> tuple[str, str, str, str]:
    parts = Path(path).parts
    if len(parts) == 3 and parts[0] == "ACPDS" and parts[1] in {"train", "valid"}:
        if not parts[2].startswith("GOPR") or Path(parts[2]).suffix != ".JPG":
            raise ValueError("Invalid ACPDS filename")
        split = parts[1]
        return (
            f"acpds-{split}-unmapped-sites",
            f"ACPDS-{split}",
            f"acpds-{split}-unmapped-session",
            "train" if split == "train" else "val",
        )
    if len(parts) != 5 or parts[1] not in VIEWS or ".." in parts:
        raise ValueError("Source outside allowlist (PUCPR/T-01 prohibited)")
    dataset, _, parking, split, _ = VIEWS[parts[1]]
    if parts[0] != dataset or Path(path).suffix != ".jpg":
        raise ValueError("Source dataset mismatch")
    return parking, parts[1], f"{parking}:{parts[3]}", split


def protected_session(path: str) -> str | None:
    parts = Path(path).parts
    if len(parts) == 5:
        if parts[0] == "CNRPark-EXT":
            return f"pisa-cnr:{parts[3]}"
        if parts[0] == "PKLot" and parts[1] in {"UFPR04", "UFPR05"}:
            return f"ufpr:{parts[3]}"
    return None


def polygon(annotation: dict, width: int, height: int) -> list[list[float]]:
    segments = annotation["segmentation"]
    if len(segments) != 1 or len(segments[0]) < 6 or len(segments[0]) % 2:
        raise ValueError("Not a single physical-slot polygon")
    points = np.array(segments[0], dtype=np.float32).reshape(-1, 2)
    if (
        not np.isfinite(points).all()
        or (points < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] >= height).any()
        or not cv2.isContourConvex(points)
        or cv2.contourArea(points) < 16
    ):
        raise ValueError("Invalid or out-of-frame polygon")
    return points.tolist()


def encode_slot(image: np.ndarray, points: list, size=224, context=0.25):
    """RGB uint8 and target mask share one crop/letterbox without cutting ROI.

    Input image is RGB. The mask marks the ground footprint; the surrounding
    context remains visible (including the elevated body of a parked vehicle).
    """
    height, width = image.shape[:2]
    xy = np.asarray(points, dtype=np.float32)
    left, top = np.floor(xy.min(axis=0)).astype(int)
    right, bottom = np.ceil(xy.max(axis=0)).astype(int) + 1
    margin = int(np.ceil(max(right - left, bottom - top) * context))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(width, right + margin), min(height, bottom + margin)
    if right <= left or bottom <= top:
        raise ValueError("Empty crop")
    rgb = image[top:bottom, left:right]
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(xy - [left, top]).astype(np.int32)], 255)
    scale = size / max(rgb.shape[:2])
    w, h = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    encoded = np.zeros((size, size, 4), dtype=np.uint8)
    encoded[:, :, :3] = [124, 116, 104]  # approximate ImageNet mean padding
    x, y = (size - w) // 2, (size - h) // 2
    encoded[y : y + h, x : x + w, :3] = rgb
    encoded[y : y + h, x : x + w, 3] = mask
    return encoded, [int(v) for v in (left, top, right, bottom)]


def archive_data(path: Path, member: str) -> dict:
    with tarfile.open(path, "r:xz") as tar:
        # No extraction of arbitrary archive paths.
        stream = tar.extractfile(member + ".json")
        if stream is None:
            raise ValueError("Missing COCO annotation member")
        data = json.load(stream)
    if {x["id"]: x["name"] for x in data["categories"]} != {0: "empty", 1: "occupied"}:
        raise ValueError("Unexpected classes; free detections are not slot labels")
    return data


def select_frames(images, excluded_sessions, per_session, seed):
    """At most N frames per view/day, at least one hour apart; no label peeking."""
    groups = defaultdict(list)
    for item in images:
        parking, camera, session, _ = source_identity(item["file_name"])
        if session not in excluded_sessions:
            groups[(parking, camera, session)].append(item)
    selected = []
    for group in sorted(groups):
        candidates = sorted(
            groups[group],
            key=lambda x: digest(f"{seed}:{x['file_name']}".encode()),
        )
        times = []
        for item in candidates:
            hour, minute, second = item["time"]
            seconds = hour * 3600 + minute * 60 + second
            if all(abs(seconds - previous) >= 3600 for previous in times):
                selected.append(item)
                times.append(seconds)
            if len(times) == per_session:
                break
    return selected


def verify(root: Path, manifest: dict, check_files=True) -> None:
    """Fail on split/group, source-annotation or file tampering before training."""
    if manifest["schema_version"] != 1 or manifest["source_commit"] != COMMIT:
        raise ValueError("Wrong manifest version")
    if manifest["recipe"] != RECIPE:
        raise ValueError("Unknown transform recipe")
    roles, seen_frames, seen_samples, seen_pixels = {}, set(), set(), set()
    originals, decoded_frames = {}, set()
    for camera, source in manifest["sources"].items():
        if camera not in VIEWS and camera not in {"ACPDS-train", "ACPDS-valid"}:
            raise ValueError("Source outside allowlist")
        if check_files:
            raw = local_path(root, source["archive"]).read_bytes()
            if digest(raw) != source["sha256"] or blob_hash(raw) != source["git_blob"]:
                raise ValueError("Annotation archive integrity failure")
            if camera in {"ACPDS-train", "ACPDS-valid"}:
                from .acpds import ANNOTATION_SHA256, official_annotations

                if digest(raw) != ANNOTATION_SHA256:
                    raise ValueError("Unpinned ACPDS annotations")
                source_split = camera.removeprefix("ACPDS-")
                if source.get("source_split") != source_split:
                    raise ValueError("ACPDS source split metadata mismatch")
                originals[camera] = official_annotations(raw, source_split)
            else:
                data = archive_data(
                    local_path(root, source["archive"]), VIEWS[camera][1]
                )
                originals[camera] = (
                    {x["id"]: x for x in data["images"]},
                    {x["id"]: x for x in data["annotations"]},
                )
    frames = {}
    for frame in manifest["frames"]:
        parking, camera, session, split = source_identity(frame["source_path"])
        if (
            frame["parking_id"],
            frame["camera_id"],
            frame["session_id"],
            frame["split"],
        ) != (parking, camera, session, split):
            raise ValueError("Wrong physical parking/camera/session split")
        if session in manifest["exclusions"]["sessions"]:
            raise ValueError("Protected session")
        for key in (f"parking:{parking}", f"camera:{camera}", f"session:{session}"):
            if roles.setdefault(key, split) != split:
                raise ValueError("Group leakage")
        if (
            frame["sha256"] in seen_frames
            or frame["sha256"] in manifest["exclusions"]["sha256"]
        ):
            raise ValueError("Duplicate or protected frame")
        seen_frames.add(frame["sha256"])
        frames[frame["source_path"]] = frame
        if check_files:
            raw = local_path(root, frame["image"]).read_bytes()
            if digest(raw) != frame["sha256"] or blob_hash(raw) != frame["git_blob"]:
                raise ValueError("Frame integrity failure")
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2] != (
                frame["height"],
                frame["width"],
            ):
                raise ValueError("Frame dimensions mismatch")
            pixel_hash = digest(decoded.tobytes() + str(decoded.shape).encode())
            if pixel_hash in decoded_frames:
                raise ValueError("Duplicate decoded frame pixels")
            decoded_frames.add(pixel_hash)
            if camera in {"ACPDS-train", "ACPDS-valid"}:
                if Path(frame["source_path"]).name not in originals[camera]:
                    raise ValueError("ACPDS image outside its official split")
            else:
                original = originals[camera][0][frame["source_image_id"]]
                if (original["file_name"], original["width"], original["height"]) != (
                    frame["source_path"],
                    frame["width"],
                    frame["height"],
                ):
                    raise ValueError("Frame differs from annotation")
    for sample in manifest["samples"]:
        frame = frames[sample["source_path"]]
        for key in ("parking_id", "camera_id", "session_id", "split"):
            if sample[key] != frame[key]:
                raise ValueError("Sample/parent leakage")
        if sample["label"] not in (0, 1) or sample["visibility"] != "unassessed":
            raise ValueError("Invalid source supervision")
        if sample["id"] in seen_samples or sample["pixel_sha256"] in seen_pixels:
            raise ValueError("Duplicate sample pixels or identity")
        seen_samples.add(sample["id"])
        seen_pixels.add(sample["pixel_sha256"])
        if check_files:
            if frame["camera_id"] in {"ACPDS-train", "ACPDS-valid"}:
                from .acpds import slot_annotation

                pair = originals[frame["camera_id"]][Path(frame["source_path"]).name]
                record = slot_annotation(frame, pair, sample["source_annotation_id"])
            else:
                record = originals[frame["camera_id"]][1][
                    sample["source_annotation_id"]
                ]
            if (
                record["image_id"] != frame["source_image_id"]
                or record["category_id"] != sample["label"]
                or polygon(record, frame["width"], frame["height"]) != sample["roi"]
            ):
                raise ValueError("Sample differs from physical-slot annotation")
            raw = local_path(root, sample["array"]).read_bytes()
            array = np.load(local_path(root, sample["array"]), allow_pickle=False)
            if (
                digest(raw) != sample["sha256"]
                or array.shape != (224, 224, 4)
                or array.dtype != np.uint8
                or digest(array.tobytes()) != sample["pixel_sha256"]
            ):
                raise ValueError("Sample integrity/format failure")
    for split in ("train", "val"):
        if {s["label"] for s in manifest["samples"] if s["split"] == split} != {0, 1}:
            raise ValueError("Each split must contain both source classes")


def prepare(workspace: Path, output: Path, per_session=2, seed=42) -> dict:
    if per_session < 1:
        raise ValueError("frames-per-session must be positive")
    previous = workspace / "paint-segmentation-provenance-2026-11-01"
    pilot = workspace / "occupancy-pilot/controlled-eval-2026-11-03-v1"
    tree_path = previous / "metapklot-immutable-tree.json"
    tree = {x["path"]: x for x in json.loads(tree_path.read_text())["tree"]}
    excluded_sessions, excluded_hashes, exclusion_sources = set(), set(), {}
    for path in sorted((workspace / "occupancy-pilot").rglob("manifest.json")):
        exclusion_sources[str(path)] = file_hash(path)
        for frame in json.loads(path.read_text()).get("frames", []):
            session = protected_session(frame.get("source_path", ""))
            if session:
                excluded_sessions.add(session)
            if "sha256" in frame:
                excluded_hashes.add(frame["sha256"])
    # Include all known source images from paint provenance, excluding their days.
    matches = json.loads((previous / "verified-source-matches.json").read_text())
    for match in matches["matches"]:
        path = match["repository_path"]
        session = protected_session(path)
        if session:
            excluded_sessions.add(session)
    # Read metadata only; never decode/open PUCPR test images/masks/annotations.
    lock = workspace / "paint-segmentation-training-2026-11-01/frozen-v1/lock.json"
    excluded_hashes.update(x["sha256"] for x in json.loads(lock.read_text())["files"])
    if (output / "manifest.json").exists():
        try:
            manifest = json.loads((output / "manifest.json").read_text())
        except json.JSONDecodeError as error:
            raise ValueError(
                "Incomplete manifest; move it aside before retrying"
            ) from error
        if manifest["seed"] != seed or manifest["frames_per_view_day"] != per_session:
            raise ValueError("Use a new output for a different preparation recipe")
        verify(output, manifest)
        print(json.dumps(manifest["counts"]), flush=True)
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_commit": COMMIT,
        "seed": seed,
        "recipe": RECIPE,
        "frames_per_view_day": per_session,
        "tree_sha256": file_hash(tree_path),
        "exclusions": {
            "sessions": sorted(excluded_sessions),
            "sha256": sorted(excluded_hashes),
            "source_manifests": exclusion_sources,
        },
        "sources": {},
        "frames": [],
        "samples": [],
        "dedup": [],
        "near_duplicate_policy": (
            "whole physical parking per split; >=1h/view/day; "
            "256-bit grayscale dHash <=4 rejected within view"
        ),
    }
    options = []
    annotations = {}
    for camera, (dataset, member, _, _, license_name) in VIEWS.items():
        relative = f"annotations/original/spots/{dataset}/{member}.tar.xz"
        # Reuse archives already fetched and verified; never download them again.
        raw = (pilot / "source" / (member + ".tar.xz")).read_bytes()
        if blob_hash(raw) != tree[relative]["sha"]:
            raise ValueError("Pinned archive mismatch")
        destination = output / "source" / (member + ".tar.xz")
        destination.write_bytes(raw)
        data = archive_data(destination, member)
        manifest["sources"][camera] = {
            "archive": str(destination.relative_to(output)),
            "sha256": digest(raw),
            "git_blob": tree[relative]["sha"],
            "url": BASE + relative,
            "license": license_name,
            "reused_from": str(pilot / "source" / destination.name),
            "available_frames": len(data["images"]),
            "available_labels": len(data["annotations"]),
        }
        annotations[camera] = defaultdict(list)
        for record in data["annotations"]:
            annotations[camera][record["image_id"]].append(record)
        candidates = [x for x in data["images"] if x["file_name"] in tree]
        options.extend(select_frames(candidates, excluded_sessions, per_session, seed))

    def fetch(item):
        path = item["file_name"]
        expected = tree[path]
        destination = output / "images" / path
        cached = pilot / "images" / path
        if destination.exists():
            raw = destination.read_bytes()
        elif cached.exists():
            raw = cached.read_bytes()
        else:
            with urlopen(BASE + path, timeout=45) as response:
                raw = response.read(expected["size"] + 1)
        if blob_hash(raw) != expected["sha"]:
            raise ValueError(f"Pinned image mismatch: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return item, raw

    seen_frames, seen_pixels, signatures = set(), {}, defaultdict(list)
    # Executor.map preserves the deterministic input order.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for item, raw in pool.map(fetch, sorted(options, key=lambda x: x["file_name"])):
            path = item["file_name"]
            parking, camera, session, split = source_identity(path)
            image_hash = digest(raw)
            if image_hash in excluded_hashes or image_hash in seen_frames:
                manifest["dedup"].append(
                    {"path": path, "reason": "exact/protected frame"}
                )
                continue
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (item["height"], item["width"]):
                raise ValueError("Image dimensions do not match annotations")
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (17, 16))
            signature = gray[:, 1:] > gray[:, :-1]
            if any(np.count_nonzero(signature != p) <= 4 for p in signatures[camera]):
                manifest["dedup"].append(
                    {"path": path, "reason": "near duplicate dHash<=4"}
                )
                continue
            signatures[camera].append(signature)
            seen_frames.add(image_hash)
            group = {
                "parking_id": parking,
                "camera_id": camera,
                "session_id": session,
                "split": split,
            }
            frame = dict(
                group,
                source_path=path,
                source_image_id=item["id"],
                width=item["width"],
                height=item["height"],
                image="images/" + path,
                sha256=image_hash,
                git_blob=tree[path]["sha"],
                source_url=BASE + path,
                date=item["date"],
                time=item["time"],
                weather=Path(path).parts[2],
            )
            manifest["frames"].append(frame)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for record in annotations[camera][item["id"]]:
                try:
                    points = polygon(record, item["width"], item["height"])
                except ValueError:
                    manifest["dedup"].append(
                        {"annotation_id": record["id"], "reason": "invalid ROI"}
                    )
                    continue
                if record["category_id"] not in (0, 1):
                    raise ValueError("Unknown label")
                array, box = encode_slot(rgb, points)
                pixel_hash = digest(array.tobytes())
                if pixel_hash in seen_pixels:
                    if seen_pixels[pixel_hash] != record["category_id"]:
                        raise ValueError("Duplicate pixels with conflicting labels")
                    manifest["dedup"].append(
                        {"annotation_id": record["id"], "reason": "duplicate crop"}
                    )
                    continue
                seen_pixels[pixel_hash] = record["category_id"]
                identity = f"{camera}-{record['id']}"
                destination = output / "arrays" / (identity + ".npy")
                np.save(destination, array, allow_pickle=False)
                roi_id = digest(json.dumps(sorted(points)).encode())[:16]
                manifest["samples"].append(
                    dict(
                        group,
                        id=identity,
                        source_path=path,
                        source_annotation_id=record["id"],
                        slot_projection_id=f"{camera}:{roi_id}",
                        physical_slot_id=None,
                        roi=points,
                        crop_box=box,
                        label=record["category_id"],
                        visibility="unassessed",
                        array=str(destination.relative_to(output)),
                        sha256=file_hash(destination),
                        pixel_sha256=pixel_hash,
                    )
                )
    from .acpds import add_validation

    add_validation(workspace, output, manifest)
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
    }
    verify(output, manifest)
    write_json(output / "manifest.partial.json", manifest)
    (output / "manifest.partial.json").replace(output / "manifest.json")
    print(json.dumps(manifest["counts"]), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-session", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output, json.loads((args.output / "manifest.json").read_text()))
    else:
        if args.workspace is None:
            parser.error("--workspace required for preparation")
        prepare(args.workspace, args.output, args.frames_per_session, args.seed)


if __name__ == "__main__":
    main()

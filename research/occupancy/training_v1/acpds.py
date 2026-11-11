"""Official ACPDS train/valid, using the already cached ZIP central directory.

MIT data declaration: Marek 2021, arXiv:2107.12207, abstract. The official
test images are never fetched, decoded, used for tuning or relabeled.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import zipfile
import zlib

import cv2
import numpy as np

from .data import (
    blob_hash,
    digest,
    encode_slot,
    file_hash,
    polygon,
    source_identity,
)

URL = "https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip"
ETAG = '"14c649b0b87549b3ab31d1c7d0ca875d-46"'
SIZE = 380796292
ANNOTATION_SHA256 = "a3c1be0dff4eb6b6ab42e64adad650f6ab245463363b8f9b58436d7d16ac0156"
INDEX_SHA256 = "337605404f88fd7d9cc7bb2b9024a2fad16a0eeddef7c46557c9cbc7bc2ec8d3"


def official_annotations(raw, split="valid"):
    if split not in {"train", "valid"}:
        raise ValueError("ACPDS test cannot be imported for training or tuning")
    source = json.loads(raw)
    selected = source[split]
    names = selected["file_names"]
    seen = set()
    for role in ("train", "valid", "test"):
        role_names = source[role]["file_names"]
        if len(set(role_names)) != len(role_names) or seen.intersection(role_names):
            raise ValueError("ACPDS official split overlap")
        seen.update(role_names)
    result = {}
    for name, rois, labels in zip(
        names, selected["rois_list"], selected["occupancy_list"], strict=True
    ):
        if len(rois) != len(labels) or not all(type(label) is bool for label in labels):
            raise ValueError("Unpaired physical-slot ROI/state")
        if (
            Path(name).name != name
            or not name.startswith("GOPR")
            or not name.endswith(".JPG")
        ):
            raise ValueError("Invalid official filename")
        result[name] = (rois, labels)
    return result


def valid_annotations(raw):
    return official_annotations(raw, "valid")


def slot_annotation(frame, pair, index):
    rois, labels = pair
    if type(index) is not int or not 0 <= index < len(rois):
        raise ValueError("Invalid source ROI index")
    points = np.asarray(rois[index], dtype=np.float64)
    if (
        points.shape != (4, 2)
        or not np.isfinite(points).all()
        or (points < 0).any()
        or (points > 1).any()
    ):
        raise ValueError("Invalid normalized source ROI")
    points *= [frame["width"] - 1, frame["height"] - 1]
    return {
        "image_id": frame["source_image_id"],
        "category_id": int(labels[index]),
        "segmentation": [points.flatten().tolist()],
    }


def fetch_range(start, end):
    # The public R2 endpoint accepts curl, but rejects urllib in this environment.
    with tempfile.TemporaryDirectory() as temporary:
        header_path = Path(temporary) / "headers"
        raw = subprocess.check_output(
            [
                "curl",
                "-fsS",
                "--retry",
                "2",
                "--max-time",
                "45",
                "--max-filesize",
                str(end - start + 1),
                "-r",
                f"{start}-{end}",
                "-D",
                str(header_path),
                URL,
            ]
        )
        headers = header_path.read_text().lower().splitlines()
    if (
        f"etag: {ETAG}" not in headers
        or f"content-range: bytes {start}-{end}/{SIZE}" not in headers
    ):
        raise ValueError("Unpinned ACPDS archive range")
    if len(raw) != end - start + 1:
        raise ValueError("Incomplete ACPDS member")
    return raw


def add_official_split(workspace, output, manifest, split="valid", cache_roots=()):
    if split not in {"train", "valid"}:
        raise ValueError("ACPDS test cannot be imported for training or tuning")
    prior = workspace / "occupancy-pilot"
    source = (
        prior / "controlled-eval-2026-11-03-v1/source/acpds-reserve/annotations.json"
    )
    zip_index = prior / "paired-acpds-2026-11-03-v1/source/index-tail.zip"
    raw = source.read_bytes()
    if digest(raw) != ANNOTATION_SHA256 or file_hash(zip_index) != INDEX_SHA256:
        raise ValueError("Cached ACPDS metadata hash mismatch")
    pairs = official_annotations(raw, split)
    destination = output / "source/acpds-annotations.json"
    if destination.exists():
        if file_hash(destination) != ANNOTATION_SHA256:
            raise ValueError("Existing ACPDS annotation copy changed")
    else:
        destination.write_bytes(raw)
    index_copy = output / "source/acpds-index-tail.zip"
    if not index_copy.exists():
        index_copy.write_bytes(zip_index.read_bytes())
    elif file_hash(index_copy) != INDEX_SHA256:
        raise ValueError("Existing ACPDS ZIP index changed")
    source_key = f"ACPDS-{split}"
    manifest["sources"][source_key] = {
        "archive": str(destination.relative_to(output)),
        "sha256": digest(raw),
        "git_blob": blob_hash(raw),
        "url": URL,
        "etag": ETAG,
        "index_sha256": INDEX_SHA256,
        "license": "MIT",
        "license_declaration": "https://arxiv.org/abs/2107.12207",
        "source_split": split,
        "reused_from": str(source),
        "grouping": (
            f"All official {split} sites/camera/sessions grouped together; "
            "physical IDs unavailable; cross-role separation follows publisher protocol"
        ),
    }
    with zipfile.ZipFile(zip_index) as archive:
        items = [archive.getinfo("images/" + name) for name in sorted(pairs)]

    def download(item):
        name = Path(item.filename).name
        dest = output / f"images/ACPDS/{split}" / name
        cached = [root / f"images/ACPDS/{split}" / name for root in cache_roots]
        if split == "train":
            cached.append(source.parent / name)
        if dest.exists():
            jpeg = dest.read_bytes()
            acquired = "already_staged"
        elif any(path.is_file() for path in cached):
            jpeg = next(path for path in cached if path.is_file()).read_bytes()
            acquired = "reused_cache"
        else:
            offset = item.header_offset + SIZE - zip_index.stat().st_size
            header = fetch_range(offset, offset + 29)
            if (
                not header.startswith(b"PK\x03\x04")
                or item.compress_type != zipfile.ZIP_DEFLATED
            ):
                raise ValueError("Unsupported ZIP member")
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            member = fetch_range(
                offset + 30, offset + 29 + name_len + extra_len + item.compress_size
            )
            if member[:name_len].decode() != item.filename:
                raise ValueError("Wrong ACPDS ZIP member")
            jpeg = zlib.decompress(member[name_len + extra_len :], -15)
            acquired = "downloaded_member"
        if len(jpeg) != item.file_size or zlib.crc32(jpeg) != item.CRC:
            raise ValueError("ACPDS ZIP CRC mismatch")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(jpeg)
        return name, jpeg, item.CRC, acquired

    seen_frames = {f["sha256"] for f in manifest["frames"]}
    seen_samples = {s["pixel_sha256"] for s in manifest["samples"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, jpeg, crc, acquired in pool.map(download, items):
            path = f"ACPDS/{split}/{name}"
            if (
                digest(jpeg) in seen_frames
                or digest(jpeg) in manifest["exclusions"]["sha256"]
            ):
                raise ValueError("ACPDS duplicate/protected frame")
            seen_frames.add(digest(jpeg))
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Invalid ACPDS JPEG")
            height, width = bgr.shape[:2]
            parking, camera, session, split = source_identity(path)
            group = {
                "parking_id": parking,
                "camera_id": camera,
                "session_id": session,
                "split": split,
            }
            frame = dict(
                group,
                source_path=path,
                source_image_id=name,
                width=width,
                height=height,
                image="images/" + path,
                sha256=digest(jpeg),
                git_blob=blob_hash(jpeg),
                zip_crc32=crc,
                source_url=URL,
                weather="unassessed",
                acquisition=acquired,
            )
            manifest["frames"].append(frame)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for index in range(len(pairs[name][0])):
                record = slot_annotation(frame, pairs[name], index)
                try:
                    points = polygon(record, width, height)
                except ValueError:
                    manifest["dedup"].append(
                        {"path": path, "roi": index, "reason": "invalid ACPDS ROI"}
                    )
                    continue
                array, box = encode_slot(rgb, points)
                pixel_hash = digest(array.tobytes())
                if pixel_hash in seen_samples:
                    raise ValueError("ACPDS duplicate crop across dataset")
                seen_samples.add(pixel_hash)
                identity = f"acpds-{split}-{name}-{index}"
                dest = output / "arrays" / (identity + ".npy")
                np.save(dest, array, allow_pickle=False)
                manifest["samples"].append(
                    dict(
                        group,
                        id=identity,
                        source_path=path,
                        source_annotation_id=index,
                        slot_projection_id=identity,
                        physical_slot_id=None,
                        roi=points,
                        crop_box=box,
                        label=record["category_id"],
                        visibility="unassessed",
                        array=str(dest.relative_to(output)),
                        sha256=file_hash(dest),
                        pixel_sha256=pixel_hash,
                    )
                )
            if len(manifest["frames"]) % 20 == 0:
                print(
                    json.dumps(
                        {"acpds_split": split, "total_frames": len(manifest["frames"])}
                    ),
                    flush=True,
                )


def add_validation(workspace, output, manifest):
    add_official_split(workspace, output, manifest, "valid")

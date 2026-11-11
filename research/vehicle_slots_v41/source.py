"""Load the exact original UFPR04 annotations and keep their source identities."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

from research.datasets_v4.contracts import Asset

COMMIT = "9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2"
UPSTREAM = "https://github.com/DSBD-Research/MetaPKLot-Dataset"
PINNED = {
    "cars": (1426408, "a208873ead0b4907ec56f601e815ecbe55d1ee05"),
    "spots": (662164, "7a613fcbc038cc029a84276f4226d7b370353fdd"),
}


@dataclass(frozen=True, slots=True)
class Bundle:
    kind: str
    archive: Asset
    git_blob: str
    json_member: str
    json_sha256: str
    upstream: str
    version: str
    license: str
    image_count: int
    annotation_count: int


@dataclass(slots=True)
class FrameSource:
    image: dict[str, Any]
    annotations: list[dict[str, Any]]


def read_bundle(
    path: Path, kind: str, image_ids: set[int]
) -> tuple[Bundle, dict[int, FrameSource]]:
    size, blob = PINNED[kind]
    path = path.resolve()
    payload = path.read_bytes()
    actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if len(payload) != size or actual != blob:
        raise ValueError(f"Unexpected bytes for the pinned original {kind} archive")
    member = f"ufpr04_{kind}.json"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
        entries = archive.getmembers()
        if len(entries) != 1 or entries[0].name != member or not entries[0].isfile():
            raise ValueError(f"Expected exactly one regular member: {member}")
        with archive.extractfile(entries[0]) as stream:
            content = stream.read()
    doc = json.loads(content)
    expected = {(1, "car")} if kind == "cars" else {(0, "empty"), (1, "occupied")}
    if {(c["id"], c["name"]) for c in doc["categories"]} != expected:
        raise ValueError(f"Unexpected original {kind} category definitions")
    frames = {}
    for image in doc["images"]:
        key = image["id"]
        if key in image_ids:
            if key in frames:
                raise ValueError(f"Duplicate {kind} image ID {key}")
            frames[key] = FrameSource(image, [])
    for annotation in doc["annotations"]:
        key = annotation["image_id"]
        if key in frames:
            frames[key].annotations.append(annotation)
        elif key in image_ids:
            raise ValueError(f"Dangling {kind} annotation image ID {key}")
    return (
        Bundle(
            kind,
            Asset(str(path), hashlib.sha256(payload).hexdigest()),
            blob,
            member,
            hashlib.sha256(content).hexdigest(),
            f"{UPSTREAM}/blob/{COMMIT}/annotations/original/{kind}/PKLot/{path.name}",
            COMMIT,
            "CC-BY-4.0 (PKLot, upstream MetaPKLot documentation)",
            len(doc["images"]),
            len(doc["annotations"]),
        ),
        frames,
    )

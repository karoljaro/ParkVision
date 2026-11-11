"""Build a diagnostic manifest and operator materials from registered originals."""

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from research.datasets_v4.contracts import (
    Asset,
    Knowledge,
    Manifest,
    Policy,
    Record,
    Registry,
    Role,
)
from research.datasets_v4.storage import fingerprint, load, save

from .overlays import render
from .provenance import Provenance
from .relations import Frame, Queue, join, rectangle
from .source import FrameSource, read_bundle

TREE_SHA256 = "56a96b43ff24df043e29e2b0eaecb55fab962fe9632cbc1aae93d7b53f6714bc"


@dataclass(frozen=True, slots=True)
class Selection:
    image_id: int
    registry_record: str


def asset(path: Path) -> Asset:
    with path.open("rb") as stream:
        return Asset(
            str(path.resolve()), hashlib.file_digest(stream, "sha256").hexdigest()
        )


def original_image(row: Record, spots: FrameSource, tree: dict) -> Image.Image:
    meta = spots.image
    if (
        meta["file_name"] != row.source_id
        or not row.source_id.startswith("PKLot/UFPR04/")
        or row.camera.value != "site:pklot-ufpr:ufpr04"
        or row.site.value != "site:pklot-ufpr"
        or meta["dataset"] != "pklot"
        or meta["subset"] != "ufpr04"
    ):
        raise ValueError("Conflicting image identity, site or camera")
    body = Path(row.asset.path).read_bytes()
    entry = tree.get(row.source_id)
    if (
        entry is None
        or len(body) != entry["size"]
        or hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        != entry["sha"]
        or hashlib.sha256(body).hexdigest() != row.asset.sha256
    ):
        raise ValueError("Image bytes disagree with the pinned upstream tree or v4")
    with Image.open(io.BytesIO(body)) as image:
        if image.size != (meta["width"], meta["height"]) or image.format != "JPEG":
            raise ValueError("Decoded image format or dimensions disagree with source")
        return image.convert("RGB")


def make_frame(
    row: Record,
    effective: Role,
    spots: FrameSource,
    cars: FrameSource | None,
    tree: dict,
    output: Path,
) -> Frame:
    meta = spots.image
    timestamp = datetime(*meta["date"], *meta["time"])
    if (
        row.day.state == Knowledge.KNOWN
        and row.day.value != timestamp.date().isoformat()
    ):
        raise ValueError("Source date contradicts the registry")
    image = original_image(row, spots, tree)
    slots, vehicles, pairs, issues = join(spots, cars, row.camera.value)
    if row.session.state != Knowledge.KNOWN:
        issues = (*issues, "session_id_unknown")
    subset = output / "source-subsets" / f"{meta['id']}.json"
    save(
        subset,
        {
            "spots_image": meta,
            "spots": spots.annotations,
            "cars_image": cars.image if cars else None,
            "cars": cars.annotations if cars else [],
            "official_jpeg_git_blob": tree[row.source_id]["sha"],
        },
    )
    frame = Frame(
        meta["id"],
        row.asset,
        row.source_id,
        *image.size,
        row.id,
        row.historical_role,
        effective,
        Role.DIAGNOSTIC,
        row.site,
        row.camera,
        row.session,
        row.day,
        timestamp.isoformat(),
        asset(subset),
        cars is not None,
        rectangle(meta),
        issues,
        slots,
        vehicles,
        pairs,
    )
    render(image, frame, spots, cars, output / "overlays")
    return frame


def write_pairs(path: Path, queue: Queue) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "image_id",
                "car_annotation_id",
                "source_car_id",
                "spot_annotation_id",
                "camera_slot_key",
                "source_link",
                "reason",
                "relation",
                "reviewer",
                "note",
            )
        )
        for frame in queue.frames:
            slots = {slot.annotation_id: slot for slot in frame.slots}
            vehicles = {vehicle.annotation_id: vehicle for vehicle in frame.vehicles}
            for pair in frame.pairs:
                slot = slots[pair.spot_annotation_id]
                writer.writerow(
                    (
                        frame.image_id,
                        pair.car_annotation_id,
                        vehicles[pair.car_annotation_id].car_id,
                        pair.spot_annotation_id,
                        slot.camera_slot_key,
                        slot.link,
                        pair.reason,
                        pair.relation,
                        pair.reviewer,
                        pair.note,
                    )
                )


def prepare(
    v4: Path,
    cars_path: Path,
    spots_path: Path,
    tree_path: Path,
    selections: tuple[Selection, ...],
    output: Path,
) -> Queue:
    if not selections or len({s.image_id for s in selections}) != len(selections):
        raise ValueError("Select unique image IDs")
    if len({s.registry_record for s in selections}) != len(selections):
        raise ValueError("Select unique v4 records")
    registry = load(v4 / "registry.json", Registry)
    policy = load(v4 / "historical-policy.json", Policy)
    curated = load(v4 / "manifest-v4.json", Manifest)
    provenance = Provenance(registry, policy, curated)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-subsets").mkdir()
    (output / "overlays").mkdir()
    tree_bytes = tree_path.read_bytes()
    if hashlib.sha256(tree_bytes).hexdigest() != TREE_SHA256:
        raise ValueError("Official UFPR04 image index does not match the pinned tree")
    tree = {entry["path"]: entry for entry in json.loads(tree_bytes)["tree"]}
    image_ids = {s.image_id for s in selections}
    car_bundle, cars = read_bundle(cars_path, "cars", image_ids)
    spot_bundle, spots = read_bundle(spots_path, "spots", image_ids)
    allowed = []
    holds = []
    for selection in selections:
        try:
            provenance.review_source(selection.registry_record)
            if selection.image_id not in spots:
                raise ValueError("Image absent from original spots")
        except (KeyError, ValueError) as exc:
            holds.append(
                {
                    "image_id": selection.image_id,
                    "registry_record": selection.registry_record,
                    "role": "quarantine",
                    "reason": str(exc),
                }
            )
        else:
            allowed.append(selection)
    manifest, validation = provenance.check(tuple(s.registry_record for s in allowed))
    save(output / "diagnostic-v4-manifest.json", manifest)
    save(output / "diagnostic-v4-validation.json", validation)
    if not validation.valid:
        raise ValueError("v4 Validation Gate rejected the diagnostic manifest")
    frames = []
    for selection in allowed:
        row, effective = provenance.review_source(selection.registry_record)
        try:
            frames.append(
                make_frame(
                    row,
                    effective,
                    spots[selection.image_id],
                    cars.get(selection.image_id),
                    tree,
                    output,
                )
            )
        except (OSError, ValueError) as exc:
            holds.append(
                {
                    "image_id": selection.image_id,
                    "registry_record": row.id,
                    "role": "quarantine",
                    "reason": str(exc),
                }
            )
    save(output / "manual-holds.json", {"records": holds})
    if len(frames) != len(allowed):
        # Publish a manifest only for the frames that actually have a queue.
        manifest, validation = provenance.check(
            tuple(f.registry_record for f in frames)
        )
        save(output / "published-v4-manifest.json", manifest)
        save(output / "published-v4-validation.json", validation)
        if not validation.valid:
            raise ValueError("v4 Validation Gate rejected the final subset")
    queue = Queue(
        provenance.registry_sha256,
        provenance.policy_sha256,
        provenance.curated_sha256,
        fingerprint(manifest),
        (car_bundle, spot_bundle),
        tuple(frames),
    )
    save(output / "queue.json", queue)
    write_pairs(output / "pairs.csv", queue)
    return queue

"""Build a standalone visual review batch from verified v4.1 pairs."""

import json
from pathlib import Path
import shutil

from PIL import Image

from research.datasets_v4.contracts import (
    Assignment,
    Manifest,
    Policy,
    Registry,
    Role,
    ValidationResult,
)
from research.datasets_v4.gate import validate
from research.datasets_v4.storage import (
    AssetVerifier,
    decode,
    encode,
    fingerprint,
    load,
    save,
)
from research.vehicle_slots_v41.prepare import asset
from research.vehicle_slots_v41.provenance import Provenance
from research.vehicle_slots_v41.relations import Queue, Relation, polygons

from .contracts import Batch, Example, PairKey, Selection
from .previews import render


def known_pairs(queue: Queue):
    return {
        PairKey(pair.image_id, pair.car_annotation_id, pair.spot_annotation_id): (
            frame,
            pair,
        )
        for frame in queue.frames
        for pair in frame.pairs
    }


def read_selection(path: Path) -> tuple[Selection, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Expected an array of selected pairs and visual suggestions")
    rows = tuple(decode(Selection, row) for row in raw)
    if not rows or len({row.pair for row in rows}) != len(rows):
        raise ValueError("The selection must contain distinct source pairs")
    return rows


def read_v41(path: Path) -> Queue:
    manifest = json.loads((path / "artifact-manifest-final.json").read_text())
    names = (
        "final/queue.json",
        "final/diagnostic-v4-manifest.json",
        "final/diagnostic-v4-validation.json",
    )
    for name in names:
        if asset(path / name).sha256 != manifest["files"][name]:
            raise ValueError(f"Historical v4.1 asset changed: {name}")
    queue = load(path / names[0], Queue)
    diagnostic = load(path / names[1], Manifest)
    gate = load(path / names[2], ValidationResult)
    if (
        queue.training_eligible
        or queue.schema_version != 1
        or not gate.valid
        or gate.issues
        or fingerprint(diagnostic) != queue.diagnostic_manifest_sha256
        or diagnostic.registry_sha256 != queue.registry_sha256
        or diagnostic.policy_sha256 != queue.policy_sha256
        or gate.registry_sha256 != queue.registry_sha256
        or gate.policy_sha256 != queue.policy_sha256
        or {a.record for a in diagnostic.assignments if a.role == Role.DIAGNOSTIC}
        != {f.registry_record for f in queue.frames}
        or {
            d.record
            for d in gate.decisions
            if d.accepted and d.effective_role == Role.DIAGNOSTIC
        }
        != {f.registry_record for f in queue.frames}
    ):
        raise ValueError("The existing v4.1 diagnostic approval could not be verified")
    return queue


def write_public(batch: Batch, output: Path) -> None:
    public = output / "public"
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(Path(__file__).with_name("static") / name, public / name)
    rows = []
    for example in batch.examples:
        row = encode(example)
        for field in ("source", "source_subset"):
            del row[field]["path"]
        row["key"] = example.pair.text
        rows.append(row)
    data = {
        "batch_sha256": fingerprint(batch),
        "examples": rows,
        "versions": [b.version for b in batch.bundles],
    }
    with (public / "batch-data.js").open("x") as stream:
        stream.write("window.PARKVISION_BATCH = ")
        stream.write(json.dumps(data, ensure_ascii=True, allow_nan=False))
        stream.write(";\n")


def prepare(v4: Path, v41: Path, selection: Path, output: Path) -> Batch:
    selected = read_selection(selection)
    queue = read_v41(v41)
    pairs = known_pairs(queue)
    frames = {}
    for row in selected:
        if row.pair not in pairs:
            raise ValueError(f"Pair absent from the v4.1 queue: {row.pair.text}")
        frame, old_pair = pairs[row.pair]
        if (
            frame.review_role != Role.DIAGNOSTIC
            or old_pair.relation != Relation.NOT_ASSESSED
            or frame.camera.value != "site:pklot-ufpr:ufpr04"
            or frame.site.value != "site:pklot-ufpr"
            or not frame.source_capture_time.startswith("2012-12-")
        ):
            raise ValueError(
                f"Expected an unassessed December UFPR04 pair: {row.pair.text}"
            )
        frames[frame.image_id] = frame
    v4_manifest = load(v4 / "manifest-v4.json", Manifest)
    authority = Provenance(
        load(v4 / "registry.json", Registry),
        load(v4 / "historical-policy.json", Policy),
        v4_manifest,
    )
    if (
        authority.registry_sha256 != queue.registry_sha256
        or authority.policy_sha256 != queue.policy_sha256
        or authority.curated_sha256 != queue.v4_manifest_sha256
    ):
        raise ValueError("The supplied v4 inputs differ from the v4.1 authority")
    for frame in frames.values():
        source, effective = authority.review_source(frame.registry_record)
        if (
            source.asset != frame.image
            or source.camera != frame.camera
            or source.site != frame.site
            or effective != frame.v4_effective_role
        ):
            raise ValueError(f"Conflicting registered source: {frame.registry_record}")
    manifest = Manifest(
        "v4.2-ab-diagnostic",
        authority.registry_sha256,
        authority.policy_sha256,
        tuple(Assignment(f.registry_record, Role.DIAGNOSTIC) for f in frames.values()),
    )
    gate = validate(authority.registry, authority.policy, manifest)
    if not gate.valid:
        raise ValueError("v4 Validation Gate rejected the selected source frames")
    verifier = AssetVerifier()
    sources = [b.archive for b in queue.bundles]
    for frame in frames.values():
        sources.extend((frame.image, frame.source_subset))
    for source in sources:
        if not verifier.verify(source):
            raise ValueError(f"Source bytes changed: {source.path}")

    output.mkdir(parents=True, exist_ok=False)
    previews = output / "public" / "images"
    previews.mkdir(parents=True)
    save(output / "diagnostic-v4-manifest.json", manifest)
    save(output / "diagnostic-v4-validation.json", gate)
    visual = {}
    for frame in frames.values():
        src = json.loads(Path(frame.source_subset.path).read_text())
        size = (frame.width, frame.height)
        spots = {s["id"]: polygons(s, size) for s in src["spots"]}
        cars = {c["id"]: polygons(c, size) for c in src["cars"]}
        original_name = f"{frame.image_id}-original.jpg"
        shutil.copyfile(frame.image.path, previews / original_name)
        with Image.open(previews / original_name) as opened:
            if opened.size != size:
                raise ValueError(f"Unexpected JPEG dimensions: {frame.image_id}")
            original = opened.convert("RGB")
        visual[frame.image_id] = original, spots, cars, f"images/{original_name}"
    examples = []
    for index, row in enumerate(selected, start=1):
        key = row.pair
        frame = frames[key.image_id]
        spot = next(s for s in frame.slots if s.annotation_id == key.spot_annotation_id)
        car = next(
            c for c in frame.vehicles if c.annotation_id == key.car_annotation_id
        )
        if not spot.camera_slot_key or not car.mask_valid or car.car_id is None:
            raise ValueError(f"Invalid source mask or slot: {key.text}")
        owners = tuple(
            s.annotation_id
            for s in frame.slots
            if car.annotation_id in s.car_annotation_ids
        )
        original, spots, cars, original_name = visual[frame.image_id]
        prefix = f"AB{index:02}"
        neighbors, full, detail_raw, detail = render(
            original,
            car.annotation_id,
            spot.annotation_id,
            cars[car.annotation_id],
            spots,
            owners,
            previews,
            prefix,
        )
        examples.append(
            Example(
                prefix,
                key,
                car.car_id,
                spot.camera_slot_key,
                frame.site.value,
                frame.camera.value,
                frame.session.value,
                frame.source_capture_time,
                frame.registry_record,
                frame.image,
                frame.source_subset,
                frame.historical_role,
                owners,
                neighbors,
                original_name,
                full,
                detail_raw,
                detail,
                row.suggestion,
            )
        )
    batch = Batch(
        asset(v41 / "final/queue.json"),
        fingerprint(queue),
        queue.registry_sha256,
        queue.policy_sha256,
        fingerprint(manifest),
        queue.bundles,
        tuple(examples),
    )
    save(output / "batch.json", batch)
    write_public(batch, output)
    return batch

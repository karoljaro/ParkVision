"""Nine original COCO exports; no implicit conversion of parking-state targets."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .contracts import (
    Annotation,
    Asset,
    Derivation,
    Meaning,
    Method,
    Record,
    Role,
    Transform,
    Verification,
)
from .import_context import ImportContext, key, lines


def annotations(names: set[str]) -> list[tuple[Meaning, tuple[str, ...]]]:
    if names == {"car"}:
        return [(Meaning.VEHICLE_MASK, ("car",))]
    if "cng" in names:
        return [(Meaning.VEHICLE_BBOX, tuple(sorted(names - {"person"})))]
    if "tree" in names:
        return [
            (Meaning.CANOPY, ("tree",)),
            (Meaning.VEHICLE_MASK, ("vehicle", "truck", "trailer", "drivingvehicle")),
            (
                Meaning.PARKING_AREA,
                ("parking lot", "parking space", "on-street parking"),
            ),
        ]
    if "Vacant" in names:
        return [
            (Meaning.SLOT_REGION, ("Occupied", "Vacant")),
            (Meaning.VEHICLE_MASK, ("vehicle",)),
        ]
    if "parkinglane" in names:
        return [(Meaning.SLOT_REGION, ("parkinglane",))]
    if "restricted-area" in names:
        return [(Meaning.PARKING_AREA, tuple(sorted(names)))]
    if "ParkingLines" in names:
        return [(Meaning.PAINT, ("ParkingLines",))]
    if "parking_space" in names:
        return [(Meaning.SLOT_REGION, tuple(sorted(names)))]
    return [(Meaning.SLOT_REGION, tuple(sorted(names)))]


def import_roboflow(ctx: ImportContext) -> None:
    staged: dict[str, Path] = {}
    for root in (
        ctx.v3 / "raw-audit",
        ctx.base / "dataset-model-improvement-2026-11-08-v1/roboflow-prepared-v1",
    ):
        for row in lines(root / "images.jsonl"):
            staged.setdefault(row["sha256"], root / row["file"])
    metadata: dict[tuple[str, str], dict] = {
        (row["archive"], row["member"]): row
        for row in lines(ctx.audit / "leakage/archive-rows.jsonl")
    }
    source_cache: dict[tuple[str, str], tuple[Asset, dict, dict]] = {}
    audit_root = ctx.audit / "technical"
    images = list(lines(audit_root / "images.jsonl"))
    profiles = {
        x["dataset"]: x
        for x in ctx.document(audit_root / "technical-report.json")["datasets"]
    }
    for row in images:
        profile = profiles[row["dataset"]]
        dataset = "rf:" + profile["sha256"][:16]
        meta = metadata[(Path(row["zip"]).name, row["member"])]
        source = Path(row["zip"])
        if dataset not in ctx.datasets:
            ev = ctx.ev(source, "/; original ZIP version and full SHA-256", Method.COCO)
            if ctx.evidence[ev].asset.sha256 != profile["sha256"]:
                raise ValueError("ZIP version differs from audited input")
            ctx.dataset(
                dataset,
                source.name,
                str(source),
                ev,
                "CC-BY-4.0 (exporter declaration, original rights chain unverified)",
                authorized=False,
                publisher="See retained original README",
            )
        dataset_ev = ctx.datasets[dataset].evidence[0]
        cache_key = (str(source), meta["annotation_member"])
        if cache_key not in source_cache:
            with zipfile.ZipFile(source) as archive:
                payload = archive.read(meta["annotation_member"])
            doc = json.loads(payload)
            names = {a["category_id"] for a in doc["annotations"]}
            classes = {c["id"]: c["name"] for c in doc["categories"]}
            active = {classes[n] for n in names}
            binding = Asset(
                str(source),
                hashlib.sha256(payload).hexdigest(),
                meta["annotation_member"],
            )
            ev_id = key("ev", str(cache_key))
            from .contracts import Evidence

            ctx.evidence[ev_id] = Evidence(
                ev_id,
                binding,
                "/images and /annotations",
                Method.COCO,
                Verification.CONFIRMED,
            )
            source_cache[cache_key] = binding, classes, {"ev": ev_id, "active": active}
        binding, classes, info = source_cache[cache_key]
        ev = info["ev"]
        asset = ctx.asset(staged[row["sha256"]], row["sha256"])
        identity = f"raw:{profile['sha256'][:16]}:{meta['split']}:{meta['image_id']}"
        if row["sha256"] not in staged:
            raise ValueError("Audited bytes are not materialized locally")
        family = "rf-family:" + key("name", dataset + ":" + meta["parent"])
        group = None
        if row["dataset"] == "vehicle-occluded":
            match = re.match(r"(IMG_.+)-\d+$", meta["original"])
            group = "road-prefix:" + match[1] if match else None
        elif row["dataset"] == "parking":
            group = "pet-mechanical-export"
        elif row["dataset"] == "parking-spaces-and-tree-canopies":
            cities = sorted(set(meta["tags"]) & {"Herne", "Paderborn"})
            group = "aerial-city:" + cities[0] if len(cities) == 1 else None
        elif row["dataset"] in {"parking-lot-line", "parking-space-detection"}:
            group = "entire-export:" + dataset
        derived = ()
        if row["dataset"] == "parking-spaces-and-tree-canopies":
            parent_id = key("aerial-parent", meta["parent"])
            tile_ev = ctx.ev(
                ctx.audit / "leakage/aerial-sampling-coverage.json",
                meta["parent"] + ": 16 spatial pieces, source parent not materialized",
                Method.VISUAL,
            )
            if parent_id not in ctx.records:
                ctx.add(
                    Record(
                        parent_id,
                        dataset,
                        meta["parent"],
                        None,
                        Verification.UNKNOWN,
                        family=ctx.fact(family, ev),
                        location_group=ctx.fact(group, ev),
                        reasons=("ORIGINAL_SPATIAL_PARENT_BYTES_UNAVAILABLE",),
                        evidence=(tile_ev,),
                    )
                )
            derived = (
                Derivation(
                    parent_id,
                    Transform.TILE,
                    (tile_ev,),
                    '{"tile_rectangle":null,"not_equivalent_augmentation":true}',
                ),
            )
        specs = []
        for meaning, source_classes in annotations(info["active"]):
            exhaustive = ("car",) if row["dataset"] == "parking" else ()
            specs.append(
                Annotation(
                    meaning,
                    source_classes,
                    exhaustive,
                    Verification.DECLARED,
                    (ev,),
                    binding,
                    f"image_id={meta['image_id']}; source semantics retained",
                )
            )
        record = Record(
            identity,
            dataset,
            meta["original"],
            asset,
            Verification.UNKNOWN,
            family=ctx.fact(family, ev),
            location_group=ctx.fact(group, ev),
            parents=derived,
            annotations=tuple(specs),
            historical_role=Role.QUARANTINE,
            training_eligible=False,
            reasons=(
                "SOURCE_PHOTOGRAPH_SITE_RIGHTS_UNRESOLVED",
                "EXPORTER_SPLIT_IS_NOT_APPROVED_ROLE:" + meta["split"],
            ),
            evidence=(dataset_ev, ev),
            rgb_size_sha256=row["pixel_sha256"],
            source_split=meta["split"],
        )
        ctx.add(record)
        ctx.technical_ids.append(identity)
        ctx.raw[(Path(row["zip"]).name, row["member"])] = identity
        ctx.original[identity] = {**meta, "technical_index": len(ctx.technical_ids) - 1}


def import_v3_views(ctx: ImportContext) -> dict[str, list[str]]:
    result = {}
    for name in ("vehicle-data", "scene-map-data"):
        path = ctx.v3 / name / "manifest.json"
        doc = ctx.document(path)
        originals = ctx.document(ctx.v3 / "raw-audit/audit.json")["archives"]
        archives = {r["sha256"][:16]: Path(r["path"]).name for r in originals}
        result[name] = []
        for index, row in enumerate(doc["samples"]):
            parent = ctx.raw[(archives[row["archive"]], row["member"])]
            ev = ctx.ev(path, f"/samples/{index}; historical role {row['split']}")
            dataset = "historical-v3:" + name
            ctx.dataset(dataset, "v3", str(path), ev)
            original = ctx.records[parent]
            identity = f"v3:{name}:{row['id']}"
            historic = Role.TRAIN if row["split"] == "train" else Role.VALIDATION
            child = replace(
                original,
                id=identity,
                dataset=dataset,
                source_id=row["id"],
                parents=(Derivation(parent, Transform.COPY, (ev,)),),
                historical_role=historic,
                training_eligible=row["training_eligible"],
                reasons=("HISTORICAL_V3_ASSIGNMENT_NOT_A_V4_APPROVAL",),
                evidence=(ev,),
            )
            ctx.add(child, lock=True)
            result[name].append(identity)
    return result

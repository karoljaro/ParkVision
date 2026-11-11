"""Native images, historical views, diagnostic sessions and visibility r4."""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import re

from .contracts import (
    Annotation,
    Derivation,
    Meaning,
    Method,
    Record,
    Role,
    Transform,
    Verification,
)
from .import_context import ImportContext, key, lines


def source_key(row: dict) -> str:
    source = row.get("source_path")
    if "acpds" in row["manifest"] or (source and source.startswith("ACPDS/")):
        return "ACPDS/" + Path(row["path"]).name.upper()
    if source and not source.startswith("data/"):
        return source
    return row["manifest"] + ":" + str(row["id"])


def role(row: dict) -> Role:
    value = row["split"]
    if value in {"test", "official test"}:
        return Role.PROTECTED_TEST
    if value == "val":
        return Role.VALIDATION
    if value in {"train", "external_train"}:
        return Role.TRAIN
    return Role.DIAGNOSTIC


def facts(row: dict) -> tuple[str | None, str | None, str | None]:
    text = row.get("source_path") or row["path"]
    parking = str(row.get("parking_id", ""))
    site = (
        "site:pklot-pucpr"
        if "/PUCPR/" in text or "PUCPR" in parking
        else (
            "site:pklot-ufpr"
            if "/UFPR" in text or parking in {"ufpr", "pklot:UFPR"}
            else (
                "site:cnr-pisa"
                if "CNRPark" in text or parking in {"pisa-cnr", "cnrpark:Pisa"}
                else None
            )
        )
    )
    match = re.search(r"(?:camera\d+|UFPR0[45])", text, re.I)
    camera = f"{site}:{match[0].lower()}" if site and match else None
    # Never parse an artifact-directory creation date as image acquisition day.
    date = re.search(r"\d{4}-\d\d-\d\d", row.get("source_path") or "")
    return site, camera, date[0] if date else None


def annotation(ctx: ImportContext, row: dict, ev: str) -> Annotation:
    manifest = ctx.base / row["manifest"]
    doc = ctx.document(manifest)
    original = doc[row["section"]][row["row_index"]]
    if row["section"] == "samples":
        kind = Meaning.PAINT
        declared = doc["classes"]
        classes = tuple(name for name in declared if name != "ignore")
        mask = manifest.parent / original["mask"]
        binding = ctx.asset(mask, original.get("mask_sha256"))
        locator = "ignore=255; source-mask mapping retained"
    else:
        kind, classes = Meaning.OCCUPANCY, ("free", "occupied")
        binding = ctx.asset(manifest)
        locator = f"/{row['section']}/{row['row_index']}; per-source ROI labels"
    return Annotation(
        kind, classes, classes, Verification.CONFIRMED, (ev,), binding, locator
    )


def import_native(ctx: ImportContext) -> None:
    rows = list(lines(ctx.audit / "leakage/external-full-image-sources.jsonl"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_key(row)].append(row)
    for source, refs in sorted(grouped.items()):
        # Prefer an actual authorised training/validation source to a later
        # diagnostic *view* of that source; retain every view separately below.
        chosen = min(
            refs,
            key=lambda r: (
                role(r) == Role.DIAGNOSTIC,
                "expansion-multisite-v1" not in r["manifest"],
                r["manifest"],
                r["row_index"],
            ),
        )
        asset = ctx.asset(Path(chosen["path"]), chosen["recorded_sha256"])
        ev = ctx.ev(
            ctx.base / chosen["manifest"], f"/{chosen['section']}/{chosen['row_index']}"
        )
        site, camera, day = facts(chosen)
        native_metadata = ctx.document(ctx.base / chosen["manifest"])[
            chosen["section"]
        ][chosen["row_index"]]
        day = native_metadata.get("capture_date") or day
        if site == "site:pklot-pucpr" and chosen.get("camera_id"):
            camera = chosen["camera_id"]
        dataset = (
            "native:acpds"
            if source.startswith("ACPDS/")
            else (
                "native:avm"
                if "avm-ss" in chosen["manifest"]
                else "native:cnr" if site == "site:cnr-pisa" else "native:pklot"
            )
        )
        version = (
            "metapklot-9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2"
            if dataset in {"native:cnr", "native:pklot"}
            else None
        )
        ctx.dataset(
            dataset,
            version,
            str(ctx.base / chosen["manifest"]),
            ev,
            (
                "MIT"
                if dataset == "native:acpds"
                else (
                    "ODbL-1.0"
                    if site == "site:cnr-pisa"
                    else "CC-BY-4.0" if site else None
                )
            ),
            authorized=bool(site) or dataset == "native:acpds",
            publisher="Original authors; attribution retained in source manifest",
        )
        license_ev = ctx.ev(
            ctx.audit / "leakage/Leakage-i-pochodzenie.md",
            "Original-source licensing and existing local academic scope",
            Method.AUDIT,
        )
        ctx.datasets[dataset] = replace(
            ctx.datasets[dataset],
            license=replace(ctx.datasets[dataset].license, evidence=(license_ev,)),
        )
        if dataset == "native:avm":
            # The pre-existing, user-accepted local academic basis is retained;
            # no broader license or redistribution right is invented.
            doc = ctx.document(ctx.base / chosen["manifest"])
            accepted = doc["provenance"]["research_use_basis_accepted"] is True
            lic = replace(
                ctx.datasets[dataset].license,
                local_research_authorized=accepted,
                verification=(
                    Verification.CONFIRMED if accepted else Verification.UNKNOWN
                ),
            )
            ctx.datasets[dataset] = replace(ctx.datasets[dataset], license=lic)
        historic = Role.PROTECTED_TEST if site == "site:pklot-pucpr" else role(chosen)
        identity = key("native", source)
        record = Record(
            identity,
            dataset,
            source,
            asset,
            Verification.CONFIRMED,
            site=ctx.fact(site, ev),
            camera=ctx.fact(camera, ev),
            day=ctx.fact(day, ev),
            family=ctx.fact(key("photo", source), ev),
            annotations=(annotation(ctx, chosen, ev),),
            historical_role=historic,
            training_eligible=historic in {Role.TRAIN, Role.VALIDATION},
            reasons=("HISTORICAL_SESSION_HINT_NOT_CONFIRMED_CAPTURE_SESSION",),
            evidence=(ev,),
            source_split=chosen["split"],
        )
        ctx.add(record, lock=True)
        ctx.by_source[source] = identity
        for row in refs:
            child_ev = ctx.ev(
                ctx.base / row["manifest"], f"/{row['section']}/{row['row_index']}"
            )
            child_asset = ctx.asset(Path(row["path"]), row["recorded_sha256"])
            child_role = (
                Role.PROTECTED_TEST if site == "site:pklot-pucpr" else role(row)
            )
            child_id = key("history", f"{row['manifest']}:{row['row_index']}")
            child = replace(
                record,
                id=child_id,
                asset=child_asset,
                parents=(
                    Derivation(
                        identity,
                        (
                            Transform.COPY
                            if asset.sha256 == child_asset.sha256
                            else Transform.UNSPECIFIED
                        ),
                        (child_ev,),
                    ),
                ),
                annotations=(annotation(ctx, row, child_ev),),
                historical_role=child_role,
                source_split=row["split"],
                training_eligible=child_role in {Role.TRAIN, Role.VALIDATION},
                evidence=(child_ev,),
            )
            ctx.add(child, lock=True)
            ctx.native_refs[(row["manifest"], row["row_index"])] = child_id


def import_sessions(ctx: ImportContext) -> None:
    index = ctx.audit / "leakage/temporal-and-visibility-image-sources.jsonl"
    for row in lines(index):
        if "visibility" in row["manifest"]:
            continue
        name = "clearpix" if "clearpix" in row["manifest"] else "family-dollar"
        ev = ctx.ev(ctx.base / row["manifest"], f"/{row['section']}/{row['row_index']}")
        dataset = "diagnostic:" + name
        ctx.dataset(dataset, "historical-session-v1", row["manifest"], ev)
        identity = f"session:{name}:{row['row_index']}"
        ctx.add(
            Record(
                identity,
                dataset,
                str(row["id"]),
                ctx.asset(Path(row["path"]), row["recorded_sha256"]),
                Verification.CONFIRMED,
                session=ctx.fact("clip:" + name, ev),
                family=ctx.fact("frame:" + name + ":" + str(row["id"]), ev),
                location_group=ctx.fact("clip:" + name, ev),
                historical_role=Role.DIAGNOSTIC,
                training_eligible=False,
                reasons=("ALREADY_EVALUATED_DIAGNOSTIC_SESSION",),
                evidence=(ev,),
            ),
            lock=True,
        )
    conf = ctx.base / "family-dollar-setup-2026-11-07-v1/approved/parking-r3.json"
    ev = ctx.ev(conf, "/; T-01 is a nonparking diagnostic region")
    parent = ctx.records["session:family-dollar:0"]
    ctx.add(
        replace(
            parent,
            id="region:family-dollar:T-01",
            source_id="R3:T-01",
            parents=(
                Derivation(
                    parent.id,
                    Transform.COPY,
                    (ev,),
                    '{"region_id":"T-01","image_carrier":"full_frame"}',
                ),
            ),
            annotations=(
                Annotation(
                    Meaning.NONPARKING,
                    ("presence",),
                    (),
                    Verification.CONFIRMED,
                    (ev,),
                    ctx.asset(conf),
                    "T-01 diagnostic projection, not a slot",
                ),
            ),
            is_parking_slot=False,
            evidence=(ev,),
            reasons=("T_01_NONPARKING_PRESENCE_ONLY",),
        ),
        lock=True,
    )
    r1 = ctx.base / "first-parking-setup-2026-11-01-v1"
    ev = ctx.ev(
        r1 / "approved/parking-r1.json",
        "/; approved configuration, image origin unresolved",
    )
    ctx.dataset("diagnostic:r1", "R1", str(r1), ev)
    ctx.add(
        Record(
            "diagnostic:R1",
            "diagnostic:r1",
            "parking.jpg",
            ctx.asset(r1 / "draft/parking.jpg"),
            Verification.UNKNOWN,
            family=ctx.fact("r1:parking.jpg", ev),
            historical_role=Role.DIAGNOSTIC,
            training_eligible=False,
            reasons=("CONFIGURATION_APPROVAL_IS_NOT_TRAINING_PROVENANCE",),
            evidence=(ev,),
        ),
        lock=True,
    )


def import_visibility(ctx: ImportContext) -> None:
    path = (
        ctx.base
        / "model-improvement-v2-2026-11-09-v1/visibility/dataset/manifest-r4.json"
    )
    doc = ctx.document(path)
    for index, row in enumerate(doc["records"]):
        ev = ctx.ev(path, f"/records/{index}; revision r4; training_eligible=false")
        ctx.dataset("visibility:r4", "r4", str(path), ev)
        ref = row["provenance"][0]
        original_path = str(Path(ref["image_path"]).resolve())
        parent = ctx.by_path.get(original_path)
        if parent is None:
            raise ValueError(
                f"Visibility source missing from native registry: {original_path}"
            )
        native = ctx.records[parent]
        frame_id = "visibility-frame:" + row["frame_pixel_sha256"]
        if frame_id not in ctx.records:
            ctx.add(
                replace(
                    native,
                    id=frame_id,
                    dataset="visibility:r4",
                    source_id=row["frame"],
                    asset=ctx.asset(path.parent / row["frame"], row["frame_sha256"]),
                    parents=(
                        Derivation(
                            parent,
                            Transform.REENCODE,
                            (ev,),
                            '{"format":"normalized_RGB_PNG"}',
                        ),
                    ),
                    annotations=(),
                    historical_role=Role.QUARANTINE,
                    training_eligible=False,
                    evidence=(ev,),
                    reasons=("VISIBILITY_V2_TRAINING_NOT_APPROVED",),
                ),
                lock=True,
            )
        excluded = "geometry_review" in row
        identity = "visibility:" + row["id"]
        ctx.add(
            replace(
                native,
                id=identity,
                dataset="visibility:r4",
                source_id=row["id"],
                asset=ctx.asset(path.parent / row["crop"], row["crop_sha256"]),
                parents=(
                    Derivation(
                        frame_id,
                        Transform.CROP,
                        (ev,),
                        json.dumps(
                            {"crop_box": row["crop_box"], "roi": row["roi_normalized"]}
                        ),
                    ),
                ),
                annotations=(
                    Annotation(
                        Meaning.VISIBILITY,
                        ("clear", "partial-but-readable", "unreadable"),
                        ("clear", "partial-but-readable", "unreadable"),
                        Verification.CONFIRMED,
                        (ev,),
                        ctx.asset(path),
                        f"/records/{index}/visibility",
                        label=row["visibility"],
                    ),
                ),
                historical_role=Role.QUARANTINE,
                training_eligible=False,
                geometry_excluded=excluded,
                evidence=(ev,),
                source_split=row["split"],
                reasons=(
                    "VISIBILITY_R4_TRAINING_ELIGIBLE_FALSE",
                    "PROPOSED_SOURCE_SPLIT:" + row["split"],
                    "GEOMETRY_EXCLUDED" if excluded else "GEOMETRY_UNASSESSED",
                ),
            ),
            lock=True,
        )

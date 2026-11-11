"""Reproducible import of the reviewed local collection; no training/deployment."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

from .contracts import (
    Assignment,
    Manifest,
    Meaning,
    Policy,
    Registry,
    Role,
    Task,
)
from .gate import validate
from .import_context import ImportContext, file_hash
from .import_history import import_native, import_sessions, import_visibility
from .import_reviews import export_candidate_queue, import_confirmed
from .import_roboflow import import_roboflow, import_v3_views
from .storage import fingerprint, save

VEHICLE_CLASSES = (
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "trailer",
    "cng",
    "rickshaw",
    "other-vehicle",
)
TASKS = (
    Task("occupancy-binary", Meaning.OCCUPANCY, ("free", "occupied")),
    Task("paint-visible", Meaning.PAINT, ("background", "paint", "curb")),
    Task("paint-binary", Meaning.PAINT, ("background", "paint")),
    Task("vehicle-instance", Meaning.VEHICLE_MASK, VEHICLE_CLASSES),
    Task("vehicle-box", Meaning.VEHICLE_BBOX, VEHICLE_CLASSES),
    Task("canopy", Meaning.CANOPY, ("tree",)),
    Task("parking-region", Meaning.PARKING_AREA, ("parking_region",)),
    Task(
        "visibility",
        Meaning.VISIBILITY,
        ("clear", "partial-but-readable", "unreadable"),
    ),
    Task("footprint", Meaning.FOOTPRINT, ("parking_slot",)),
)


def manifest(
    registry: Registry, policy: Policy, identity: str, assignments: list[Assignment]
) -> Manifest:
    return Manifest(
        identity, fingerprint(registry), fingerprint(policy), tuple(assignments), TASKS
    )


def compile_local(base: Path, audit: Path, v3: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    ctx = ImportContext(base.resolve(), audit.resolve(), v3.resolve(), output.resolve())
    snapshot = output / "executed-code"
    snapshot.mkdir()
    code = {}
    for source in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(source, snapshot / source.name)
        code[source.name] = file_hash(source)
    save(
        output / "code-version.json",
        {
            "source_sha256": code,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
            ).strip(),
            "warning": "Working-tree SHA-256 is separate from HEAD",
        },
    )
    ctx.verify_audit()
    import_native(ctx)
    import_sessions(ctx)
    import_visibility(ctx)
    import_roboflow(ctx)
    confirmed = import_confirmed(ctx)
    views = import_v3_views(ctx)
    candidate_asset, candidates = export_candidate_queue(ctx, confirmed)
    registry = Registry(
        "parkvision-dataset-registry-v4.0",
        tuple(ctx.datasets.values()),
        tuple(ctx.records.values()),
        tuple(ctx.evidence.values()),
        tuple(ctx.relations.values()),
        tuple(ctx.reviewed.values()),
        candidate_indices=(candidate_asset,),
    )
    protected = tuple(
        sorted(
            {
                row.asset.sha256
                for row in registry.records
                if row.historical_role == Role.PROTECTED_TEST and row.asset
            }
        )
    )
    policy = Policy(
        "parkvision-historical-boundaries-v4.0",
        tuple(ctx.locks.values()),
        ("site:pklot-pucpr",),
        ("site:cnr-pisa",),
        protected,
        tuple(ctx.evidence.values()),
    )
    save(output / "registry.json", registry)
    save(output / "historical-policy.json", policy)
    requested = []
    for row in registry.records:
        if row.asset is None:
            continue  # Metadata-only spatial parents remain in the registry.
        role = row.historical_role
        # Canonical native images and original manual paint targets can be
        # considered, but unknown sites/eligibility still go through the gate.
        if row.id.startswith("native:"):
            task = (
                (
                    "paint-visible"
                    if "curb" in row.annotations[0].classes
                    else "paint-binary"
                )
                if row.annotations[0].meaning == Meaning.PAINT
                else "occupancy-binary"
            )
        elif (
            row.id.startswith("history:")
            and row.annotations[0].meaning == Meaning.PAINT
            and not (
                row.parents
                and ctx.records[row.parents[0].parent].annotations == row.annotations
            )
        ):
            task = "paint-visible"
        else:
            task = None
            if role in {Role.TRAIN, Role.VALIDATION, Role.TEST}:
                role = Role.QUARANTINE
        requested.append(Assignment(row.id, role, task))
    proposed = manifest(registry, policy, "v4.0-import-review", requested)
    save(output / "requested-manifest.json", proposed)
    result = validate(registry, policy, proposed)
    save(output / "import-validation.json", result)
    effective = [
        replace(a, role=d.effective_role)
        for a, d in zip(requested, result.decisions, strict=True)
    ]
    curated = manifest(registry, policy, "v4.0-manifest", effective)
    save(output / "manifest-v4.json", curated)
    verified = validate(registry, policy, curated)
    save(output / "manifest-v4-validation.json", verified)
    if not verified.valid:
        raise ValueError(
            "Curated output failed validation; see manifest-v4-validation.json"
        )
    save(
        output / "quarantine.json",
        {
            "records": [
                {
                    "record": row.id,
                    "historical_role": row.historical_role.value,
                    "evidence": row.evidence,
                    "source_reasons": row.reasons,
                    "validation_rules": d.reasons,
                }
                for d in result.decisions
                if d.effective_role == Role.QUARANTINE
                for row in [ctx.records[d.record]]
            ],
            "semantics": "Explicit quarantined records, not approved training data",
        },
    )
    v3_assignments = []
    for identity in views["vehicle-data"]:
        row = ctx.records[identity]
        meaning = row.annotations[0].meaning
        task = "vehicle-box" if meaning == Meaning.VEHICLE_BBOX else "vehicle-instance"
        v3_assignments.append(Assignment(identity, row.historical_role, task))
    legacy = manifest(registry, policy, "v3-vehicle-assignment-audit", v3_assignments)
    save(output / "v3-vehicle-manifest-view.json", legacy)
    legacy_result = validate(registry, policy, legacy)
    save(output / "v3-vehicle-validation.json", legacy_result)
    road = [
        a
        for a in v3_assignments
        if ctx.records[a.record].location_group.value
        and ctx.records[a.record].location_group.value.startswith("road-prefix:")
    ]
    road_proposal = manifest(
        registry,
        policy,
        "v4.0-road-candidate-train-only",
        [replace(a, role=Role.TRAIN) for a in road],
    )
    save(output / "road-proposal-manifest.json", road_proposal)
    road_result = validate(registry, policy, road_proposal)
    save(output / "road-proposal-validation.json", road_result)
    save(
        output / "road-proposal.json",
        {
            "confirmed_dependency_group": [
                "IMG_0005",
                "IMG_0009",
                "IMG_0380",
                "IMG_1084",
            ],
            "candidate_assignment": "train-only; provenance and coverage pending",
            "counterfactual_only": True,
            "historical_roles_remain_locked": True,
            "independent_validation_available": False,
            "approved_for_training": False,
            "current_v3_changed": False,
            "all_road_records": len(road),
            "reason": "No independent route is established for other prefixes",
        },
    )
    stats = {
        "records": len(registry.records),
        "datasets": len(registry.datasets),
        "evidence": len(registry.evidence),
        "confirmed_relations": len(registry.relations),
        "pending_candidates": candidates,
        "spatial_parents_without_bytes": sum(
            r.id.startswith("aerial-parent:") and r.asset is None
            for r in registry.records
        ),
        "registry_sha256": fingerprint(registry),
        "policy_sha256": fingerprint(policy),
        "source_counts": dict(Counter(row.dataset for row in registry.records)),
        "effective_roles": dict(
            Counter(d.effective_role.value for d in result.decisions)
        ),
        "approved_learning": sum(
            d.accepted and d.effective_role in {Role.TRAIN, Role.VALIDATION}
            for d in result.decisions
        ),
        "v3_location_conflicts": sum(
            i.rule == "LOCATION_OR_FAMILY_SPLIT" for i in legacy_result.issues
        ),
        "collection_training_ready": False,
        "AB_labels_created": False,
        "historical_data_changed": False,
    }
    save(output / "summary.json", stats)
    save(output / "inputs.json", ctx.inputs)
    return stats

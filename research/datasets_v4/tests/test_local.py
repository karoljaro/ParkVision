"""Acceptance against the local frozen sources, not generated image fixtures."""

from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from research.datasets_v4.compiler import manifest
from research.datasets_v4.contracts import (
    Assignment,
    Candidate,
    Knowledge,
    Manifest,
    Meaning,
    Policy,
    Registry,
    RelationKind,
    ReviewState,
    Role,
    Transform,
    ValidationResult,
)
from research.datasets_v4.gate import validate
from research.datasets_v4.graph import build_graph
from research.datasets_v4.storage import decode, load, save

ROADS = {"IMG_0005", "IMG_0009", "IMG_0380", "IMG_1084"}
LOCAL = os.environ.get("PARKVISION_V4_ARTIFACTS")


@pytest.mark.skipif(not LOCAL, reason="Set PARKVISION_V4_ARTIFACTS to a local build")
def test_reviewed_local_collection():
    root = Path(LOCAL)
    registry = load(root / "registry.json", Registry)
    policy = load(root / "historical-policy.json", Policy)
    curated = load(root / "manifest-v4.json", Manifest)
    current = load(root / "v3-vehicle-manifest-view.json", Manifest)
    old_v3 = load(root / "v3-vehicle-validation.json", ValidationResult)
    final = load(root / "manifest-v4-validation.json", ValidationResult)
    assert final.valid
    graph, structural = build_graph(registry, policy)
    assert not structural
    rows = {r.id: r for r in registry.records}
    roles = {a.record: a.role for a in curated.assignments}
    requested = []
    cases = {}

    for number, image, transform in (
        (474, "GOPR6536", Transform.ROTATION),
        (515, "GOPR6721", Transform.REENCODE),
    ):
        key = f"raw:63bf7906d03be257:train:{number}"
        child = rows[key]
        [parent] = [p for p in child.parents if p.transform == transform]
        assert image in rows[parent.parent].source_id
        assert rows[parent.parent].historical_role == Role.PROTECTED_TEST
        assert roles[parent.parent] == Role.PROTECTED_TEST
        assert rows[parent.parent].asset.sha256 in policy.protected_assets
        cases[image] = {
            "record": key,
            "protected_parent": parent.parent,
            "transform": transform,
        }
        requested.append(Assignment(key, Role.TRAIN, "occupancy-binary"))

    holdout = [
        r
        for r in rows.values()
        if r.id.startswith("native:") and r.historical_role == Role.PROTECTED_TEST
    ]
    protected = [r for r in rows.values() if r.historical_role == Role.PROTECTED_TEST]
    assert len(holdout) == 30 and len(protected) == 60
    assert all(roles[r.id] == Role.PROTECTED_TEST for r in protected)
    pucpr = "raw:63bf7906d03be257:train:1353"
    assert rows[pucpr].day.value == "2012-11-06"
    assert rows[pucpr].site.value == "site:pklot-pucpr"
    frozen_paint = [r for r in holdout if r.site.value == rows[pucpr].site.value]
    assert len(frozen_paint) == 3
    assert all(r.day.value != rows[pucpr].day.value for r in frozen_paint)
    assert all(graph.photo.find(r.id) != graph.photo.find(pucpr) for r in frozen_paint)
    requested.append(Assignment(pucpr, Role.TRAIN, "paint-visible"))

    road = [
        a
        for a in current.assignments
        if (rows[a.record].location_group.value or "").startswith("road-prefix:")
    ]
    assert Counter(a.role for a in road) == {Role.TRAIN: 512, Role.VALIDATION: 80}
    road_split = [
        issue for issue in old_v3.issues if issue.rule == "LOCATION_OR_FAMILY_SPLIT"
    ]
    [conflict] = road_split
    assert not old_v3.valid and len(conflict.records) == 163
    assert len({graph.dependency.find(key) for key in conflict.records}) == 1
    assert {
        rows[key].location_group.value.split(":")[1] for key in conflict.records
    } == ROADS
    road_pairs = [
        rel.id
        for rel in registry.relations
        if rel.kind == RelationKind.LOCATION_DEPENDENCY
        and rel.left.startswith("raw:c31b9c709e97d847:")
        and rel.right.startswith("raw:c31b9c709e97d847:")
    ]
    assert len(road_pairs) == 4
    requested.extend(road)

    tiles = defaultdict(list)
    for row in rows.values():
        for edge in row.parents:
            if edge.transform == Transform.TILE:
                tiles[edge.parent].append(row)
    assert len(tiles) == 25 and all(len(children) == 16 for children in tiles.values())
    parent, siblings = next(
        (key, children)
        for key, children in tiles.items()
        if len({r.asset.sha256 for r in children}) == 16
        and len({graph.content.find(r.id) for r in children}) == 16
    )
    assert rows[parent].asset is None
    assert len({graph.dependency.find(r.id) for r in siblings}) == 1
    tile_a, tile_b = siblings[:2]
    requested.extend(
        (
            Assignment(tile_a.id, Role.TRAIN, "parking-region"),
            Assignment(tile_b.id, Role.VALIDATION, "parking-region"),
        )
    )

    unknown = next(
        r.id
        for r in rows.values()
        if r.dataset == "rf:44f51e03ba1cd4e8" and r.site.state == Knowledge.UNKNOWN
    )
    assert roles[unknown] == Role.QUARANTINE
    requested.append(Assignment(unknown, Role.TRAIN, "footprint"))

    pet = "v3:vehicle-data:d73481d9cfe9d2a0:test:123"
    assert rows[pet].historical_role == Role.TRAIN
    assert rows[pet].annotations[0].classes == ("car",)
    requested.append(Assignment(pet, Role.TRAIN, "vehicle-instance"))

    excluded_keys = {a.record for a in requested}
    with Path(registry.candidate_indices[0].path).open() as stream:
        for line in stream:
            candidate = decode(Candidate, json.loads(line))
            if {candidate.left, candidate.right} & excluded_keys:
                continue
            if not all(
                key.startswith("raw:") for key in (candidate.left, candidate.right)
            ):
                continue
            if graph.photo.find(candidate.left) == graph.photo.find(candidate.right):
                continue
            if graph.dependency.find(candidate.left) != graph.dependency.find(
                candidate.right
            ):
                break
        else:
            pytest.fail("No genuine unresolved similarity candidate found")
    assert candidate.status == ReviewState.PENDING
    requested.append(Assignment(candidate.left, Role.TRAIN, "vehicle-instance"))
    assert (
        len([c for c in registry.candidates if c.status == ReviewState.REJECTED]) == 2
    )

    t01 = "region:family-dollar:T-01"
    assert rows[t01].is_parking_slot is False
    assert rows[t01].annotations[0].meaning == Meaning.NONPARKING
    assert roles[t01] == Role.DIAGNOSTIC
    requested.append(Assignment(t01, Role.TRAIN, "footprint"))
    cnr = next(
        r.id
        for r in rows.values()
        if r.site.value == "site:cnr-pisa"
        and r.historical_role == Role.VALIDATION
        and r.annotations[0].meaning == Meaning.PAINT
    )
    requested.append(Assignment(cnr, Role.TRAIN, "paint-visible"))

    visibility = [r for r in rows.values() if r.id.startswith("visibility:")]
    source_path = Path(
        next(d.source for d in registry.datasets if d.id == "visibility:r4")
    )
    original = {r["id"]: r for r in json.loads(source_path.read_text())["records"]}
    assert len(visibility) == len(original) == 196
    assert sum(r.geometry_excluded for r in visibility) == 14
    for row in visibility:
        assert row.annotations[0].label == original[row.source_id]["visibility"]
        assert row.source_split == original[row.source_id]["split"]
        assert row.geometry_excluded == ("geometry_review" in original[row.source_id])
        assert original[row.source_id]["training_eligible"] is False
        assert roles[row.id] == Role.QUARANTINE and not row.training_eligible
        requested.append(Assignment(row.id, Role.TRAIN, "visibility"))
    requested.extend(Assignment(r.id, Role.TRAIN, "paint-visible") for r in holdout)
    probe = manifest(registry, policy, "v4.0-real-data-acceptance", requested)
    result = validate(registry, policy, probe)
    assert not result.valid
    decisions = {d.record: d for d in result.decisions}

    def blocked(key, *rules):
        assert decisions[key].effective_role == Role.QUARANTINE
        assert not decisions[key].accepted
        assert set(rules) <= set(decisions[key].reasons)
        return list(rules)

    for image in ("GOPR6536", "GOPR6721"):
        cases[image]["rules"] = blocked(
            cases[image]["record"],
            "PROTECTED_TEST_DERIVATIVE",
            "PROTECTED_SOURCE_DEPENDENCY",
        )
    validation_probe = replace(
        probe,
        id="v4.0-real-pucpr-validation-probe",
        assignments=(Assignment(pucpr, Role.VALIDATION, "paint-visible"),),
    )
    against_validation = validate(registry, policy, validation_probe)
    assert not against_validation.valid
    assert "PROTECTED_PHYSICAL_SITE" in against_validation.decisions[0].reasons
    cases["PUCPR"] = {
        "record": pucpr,
        "day": rows[pucpr].day.value,
        "protected_days": sorted({r.day.value for r in frozen_paint}),
        "train_rules": blocked(pucpr, "PROTECTED_PHYSICAL_SITE"),
        "validation_rules": against_validation.decisions[0].reasons,
    }
    for key in conflict.records:
        blocked(key, "LOCATION_OR_FAMILY_SPLIT")
    cases["road"] = {
        "groups": sorted(ROADS),
        "confirmed_relation_ids": road_pairs,
        "train_validation_total": [512, 80],
        "conflict_train_validation": dict(
            Counter(decisions[key].requested_role for key in conflict.records)
        ),
        "rule": "LOCATION_OR_FAMILY_SPLIT",
    }
    cases["aerial"] = {
        "parent": parent,
        "tile_records": [r.id for r in siblings],
        "rules": blocked(tile_a.id, "LOCATION_OR_FAMILY_SPLIT"),
    }
    blocked(tile_b.id, "LOCATION_OR_FAMILY_SPLIT")
    cases["unknown"] = {
        "record": unknown,
        "rules": blocked(unknown, "UNCONFIRMED_PROVENANCE", "UNKNOWN_PHYSICAL_SITE"),
    }
    cases["PET"] = {
        "record": pet,
        "rules": blocked(pet, "INCOMPLETE_CLASS_COVERAGE"),
    }
    cases["similarity"] = {
        "candidate": candidate.id,
        "endpoints": [candidate.left, candidate.right],
        "joined": False,
        "rules": blocked(candidate.left, "SIMILARITY_REVIEW_REQUIRED"),
    }
    cases["T-01"] = {
        "record": t01,
        "rules": blocked(t01, "NOT_A_PARKING_SLOT", "DIAGNOSTIC_PROMOTION"),
    }
    blocked(cnr, "VALIDATION_ONLY_SITE")
    for row in holdout:
        blocked(row.id, "PROTECTED_TEST_DERIVATIVE")
    for row in visibility:
        blocked(row.id, "NOT_TRAINING_ELIGIBLE")
        if row.geometry_excluded:
            blocked(row.id, "GEOMETRY_EXCLUDED")
    cases["history_and_visibility"] = {
        "protected_originals": len(holdout),
        "protected_records": len(protected),
        "visibility_records": len(visibility),
        "visibility_labels": dict(Counter(r.annotations[0].label for r in visibility)),
        "geometry_exclusions": 14,
        "cnr": {"record": cnr, "rule": "VALIDATION_ONLY_SITE"},
    }
    assert len(cases) == 10
    if output := os.environ.get("PARKVISION_V4_ACCEPTANCE_OUTPUT"):
        path = Path(output)
        save(path, {"status": "passed", "cases": cases})
        save(path.with_name("acceptance-manifest.json"), probe)
        save(path.with_name("acceptance-validation.json"), result)
        save(path.with_name("acceptance-pucpr-val-manifest.json"), validation_probe)
        save(path.with_name("acceptance-pucpr-val-validation.json"), against_validation)

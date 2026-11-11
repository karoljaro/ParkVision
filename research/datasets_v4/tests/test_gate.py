"""Behavioral contracts: lineage, protected roles, geometry and class coverage."""

from dataclasses import replace

import pytest

from research.datasets_v4.contracts import (
    Annotation,
    Approval,
    Asset,
    Assignment,
    Candidate,
    Dataset,
    Derivation,
    Evidence,
    Fact,
    Knowledge,
    License,
    Lock,
    Manifest,
    Meaning,
    Method,
    Policy,
    Protocol,
    Record,
    Registry,
    Relation,
    RelationKind,
    Role,
    Task,
    Transform,
    Verification,
)
from research.datasets_v4.gate import validate
from research.datasets_v4.graph import build_graph
from research.datasets_v4.import_context import file_hash
from research.datasets_v4.storage import decode, encode, fingerprint, load, save


@pytest.fixture
def example(tmp_path):
    def asset(name, text):
        p = tmp_path / name
        p.write_text(text)
        return Asset(str(p), file_hash(p))

    proof = asset("provenance.json", '{"review":"explicit source metadata"}')
    ev = Evidence(
        "ev:source", proof, "/review", Method.MANIFEST, Verification.CONFIRMED
    )
    refs = (ev.id,)
    dataset = Dataset(
        "data:one",
        "release-1",
        "local-reviewed-source",
        License("local-test-contract", "fixture", Verification.CONFIRMED, True, refs),
        refs,
    )
    label = asset("annotations.json", '{"class":"car"}')
    anno = Annotation(
        Meaning.VEHICLE_MASK,
        ("car",),
        ("car",),
        Verification.CONFIRMED,
        refs,
        label,
        "/class",
    )

    def record(identity="image:a", site="site:a", image_text="a"):
        return Record(
            identity,
            dataset.id,
            identity,
            asset(identity.replace(":", "-") + ".bin", image_text),
            Verification.CONFIRMED,
            site=Fact(Knowledge.KNOWN, site, refs),
            family=Fact(Knowledge.KNOWN, "family:" + identity, refs),
            annotations=(anno,),
            historical_role=Role.TRAIN,
            training_eligible=True,
            evidence=refs,
        )

    row = record()
    registry = Registry("registry", (dataset,), (row,), (ev,))
    policy = Policy("trusted-policy", (), (), (), (), ())
    task = Task("cars", Meaning.VEHICLE_MASK, ("car",))

    def check(reg=registry, pol=policy, assignments=None, tasks=(task,)):
        assignments = assignments or (Assignment(row.id, Role.TRAIN, task.id),)
        manifest = Manifest(
            "manifest",
            fingerprint(reg),
            fingerprint(pol),
            tuple(assignments),
            tuple(tasks),
        )
        return validate(reg, pol, manifest)

    return registry, policy, row, refs, record, check, asset


def rules(result):
    return {issue.rule for issue in result.issues}


def test_confirmed_sources_from_different_sites_can_use_different_splits(example):
    reg, pol, first, _, record, check, _ = example
    second = record("image:b", "site:b", "b")
    reg = replace(reg, records=(first, second))
    result = check(
        reg,
        pol,
        (
            Assignment(first.id, Role.TRAIN, "cars"),
            Assignment(second.id, Role.VALIDATION, "cars"),
        ),
    )
    assert result.valid and all(d.accepted for d in result.decisions)


def test_same_site_across_cameras_or_days_is_not_independent(example):
    reg, pol, first, refs, record, check, _ = example
    second = replace(
        record("image:b", "site:a", "b"),
        camera=Fact(Knowledge.KNOWN, "different-camera", refs),
        day=Fact(Knowledge.KNOWN, "2026-11-11", refs),
    )
    result = check(
        replace(reg, records=(first, second)),
        pol,
        (
            Assignment(first.id, Role.TRAIN, "cars"),
            Assignment(second.id, Role.TEST, "cars"),
        ),
    )
    assert "LOCATION_OR_FAMILY_SPLIT" in rules(result)


@pytest.mark.parametrize("transform", list(Transform))
def test_protected_photo_inherits_through_every_declared_transform(example, transform):
    reg, pol, first, refs, record, check, _ = example
    protected = replace(
        first, historical_role=Role.PROTECTED_TEST, training_eligible=False
    )
    derived = replace(
        record("image:derived", "site:a", "derived"),
        parents=(Derivation(first.id, transform, refs),),
    )
    reg = replace(reg, records=(protected, derived))
    pol = replace(pol, locks=(Lock(first.id, Role.PROTECTED_TEST, refs, False),))
    result = check(reg, pol, (Assignment(derived.id, Role.TRAIN, "cars"),))
    assert "PROTECTED_TEST_DERIVATIVE" in rules(result)
    assert result.decisions[0].effective_role == Role.QUARANTINE


def test_photo_under_another_name_and_encoding_remains_protected(example):
    reg, pol, first, refs, record, check, _ = example
    protected = replace(
        first, historical_role=Role.PROTECTED_TEST, training_eligible=False
    )
    alias = record("unrelated-name:42", "site:a", "reencoded")
    relation = Relation(
        "review:identity", first.id, alias.id, RelationKind.SAME_PHOTOGRAPH, refs
    )
    reg = replace(reg, records=(protected, alias), relations=(relation,))
    result = check(reg, pol, (Assignment(alias.id, Role.TRAIN, "cars"),))
    assert "PROTECTED_TEST_DERIVATIVE" in rules(result)


def test_unreviewed_similarity_is_queued_without_union(example):
    reg, pol, first, refs, record, check, _ = example
    second = record("image:b", "site:b", "b")
    candidate = Candidate("possible:match", first.id, second.id, "pHash=0", refs)
    reg = replace(reg, records=(first, second), candidates=(candidate,))
    graph, errors = build_graph(reg, pol)
    assert not errors and graph.photo.find(first.id) != graph.photo.find(second.id)
    result = check(
        reg,
        pol,
        (
            Assignment(first.id, Role.TRAIN, "cars"),
            Assignment(second.id, Role.VALIDATION, "cars"),
        ),
    )
    assert "SIMILARITY_REVIEW_REQUIRED" in rules(result)
    assert not rules(result) & {"PHOTOGRAPH_SPLIT", "LOCATION_OR_FAMILY_SPLIT"}


def test_unknown_is_not_confirmed_absence_and_neither_is_a_known_site(example):
    reg, pol, row, refs, _, check, _ = example
    for site in (Fact(), Fact(Knowledge.ABSENT, None, refs)):
        result = check(replace(reg, records=(replace(row, site=site),)), pol)
        assert "UNKNOWN_PHYSICAL_SITE" in rules(result)
    with pytest.raises(ValueError):
        Fact(Knowledge.UNKNOWN, "fake-site")


def test_car_only_is_not_exhaustive_vehicle_supervision(example):
    reg, pol, _, _, _, check, _ = example
    task = Task("vehicles", Meaning.VEHICLE_MASK, ("car", "motorcycle", "bus"))
    result = check(reg, pol, (Assignment("image:a", Role.TRAIN, task.id),), (task,))
    assert "INCOMPLETE_CLASS_COVERAGE" in rules(result)


@pytest.mark.parametrize(
    "source,classes",
    [
        (Meaning.SLOT_REGION, ("Vacant",)),
        (Meaning.PARKING_AREA, ("free-space-parking",)),
        (Meaning.SLOT_REGION, ("parkinglane",)),
        (Meaning.VEHICLE_BBOX, ("car",)),
    ],
)
def test_targets_are_not_implicitly_reinterpreted(example, source, classes):
    reg, pol, row, _, _, check, _ = example
    row = replace(
        row,
        is_parking_slot=True,
        annotations=(
            replace(
                row.annotations[0],
                meaning=source,
                classes=classes,
                exhaustive_classes=classes,
            ),
        ),
    )
    task = Task("physical", Meaning.FOOTPRINT, ("parking_slot",))
    result = check(
        replace(reg, records=(row,)),
        pol,
        (Assignment(row.id, Role.TRAIN, task.id),),
        (task,),
    )
    assert "ANNOTATION_SEMANTICS" in rules(result)


def test_explicit_scoped_semantic_approval_requires_artifact_and_implementation(
    example,
):
    reg, pol, row, refs, _, check, asset = example
    row = replace(
        row,
        is_parking_slot=True,
        annotations=(replace(row.annotations[0], meaning=Meaning.SLOT_REGION),),
    )
    operator = replace(reg.evidence[0], id="ev:operator", method=Method.OPERATOR)
    approval = Approval(
        "approval:one",
        (row.id,),
        Meaning.SLOT_REGION,
        Meaning.FOOTPRINT,
        ("parking_slot",),
        Protocol.SEMANTIC_TRANSFORM,
        "reviewer",
        "New ground polygons reviewed",
        asset("reviewed-labels.json", "new physical polygons"),
        asset("protocol.txt", "operator reviewed transform"),
        (operator.id,),
    )
    reg = replace(
        reg, records=(row,), evidence=(*reg.evidence, operator), approvals=(approval,)
    )
    task = Task("physical", Meaning.FOOTPRINT, ("parking_slot",), approval.id)
    assert check(reg, pol, (Assignment(row.id, Role.TRAIN, task.id),), (task,)).valid
    broken = replace(approval, artifact=replace(approval.artifact, sha256="0" * 64))
    result = check(
        replace(reg, approvals=(broken,)),
        pol,
        (Assignment(row.id, Role.TRAIN, task.id),),
        (task,),
    )
    assert "INVALID_TASK_APPROVAL" in rules(result)


@pytest.mark.parametrize("bad_parent", ["missing:parent", "cycle"])
def test_missing_parent_and_cycle_fail_closed(example, bad_parent):
    reg, pol, row, refs, record, check, _ = example
    parent = row.id if bad_parent == "cycle" else bad_parent
    reg = replace(
        reg,
        records=(replace(row, parents=(Derivation(parent, Transform.CROP, refs),)),),
    )
    result = check(reg, pol)
    assert ("LINEAGE_CYCLE" if bad_parent == "cycle" else "MISSING_PARENT") in rules(
        result
    )
    assert not result.valid


def test_false_visibility_eligibility_and_nonparking_geometry_cannot_be_overwritten(
    example,
):
    reg, pol, row, refs, _, check, _ = example
    diagnostic = replace(
        row,
        historical_role=Role.DIAGNOSTIC,
        training_eligible=False,
        geometry_excluded=True,
        is_parking_slot=False,
    )
    pol = replace(pol, locks=(Lock(row.id, Role.DIAGNOSTIC, refs, False, True, False),))
    changed = replace(
        diagnostic,
        historical_role=Role.TRAIN,
        training_eligible=True,
        geometry_excluded=False,
        is_parking_slot=True,
    )
    result = check(replace(reg, records=(changed,)), pol)
    assert "HISTORICAL_METADATA_CHANGED" in rules(result)
    assert "DIAGNOSTIC_PROMOTION" in rules(result)


def test_modified_bytes_and_duplicate_assignments_are_errors(example):
    reg, pol, row, _, _, check, _ = example
    from pathlib import Path

    Path(row.asset.path).write_text("changed")
    result = check(
        reg,
        pol,
        (
            Assignment(row.id, Role.TRAIN, "cars"),
            Assignment(row.id, Role.VALIDATION, "cars"),
        ),
    )
    assert {"IMAGE_INTEGRITY", "DUPLICATE_ASSIGNMENT"} <= rules(result)


def test_separate_trusted_policy_hash_cannot_be_silently_replaced(example):
    reg, pol, row, _, _, _, _ = example
    manifest = Manifest(
        "manifest",
        fingerprint(reg),
        "0" * 64,
        (Assignment(row.id, Role.TRAIN, "cars"),),
    )
    result = validate(reg, pol, manifest)
    assert "POLICY_HASH_MISMATCH" in rules(result)


@pytest.mark.parametrize(
    "mutation", ["extra", "invalid-id", "bool-version", "unknown-role"]
)
def test_strict_json_boundary(example, mutation):
    reg, pol, row, _, _, _, _ = example
    value = encode(reg)
    if mutation == "extra":
        value["fallback"] = True
    if mutation == "invalid-id":
        value["records"][0]["id"] = "../bad"
    if mutation == "bool-version":
        value["schema_version"] = True
    if mutation == "unknown-role":
        value["records"][0]["historical_role"] = "approved-ish"
    with pytest.raises((ValueError, TypeError)):
        decode(Registry, value)


def test_duplicate_json_keys_are_not_last_value_wins(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"id":"first","id":"second"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load(path, Registry)


def test_roundtrip_preserves_typed_roles_and_unknown_facts(example, tmp_path):
    reg, _, _, _, _, _, _ = example
    path = tmp_path / "registry.json"
    save(path, reg)
    restored = load(path, Registry)
    assert restored == reg and restored.records[0].camera.state is Knowledge.UNKNOWN


def test_protected_role_cannot_be_assigned_as_bypass(example):
    reg, pol, row, _, _, check, _ = example
    result = check(reg, pol, (Assignment(row.id, Role.PROTECTED_TEST),))
    assert not result.valid and "UNAUTHORIZED_PROTECTED_ROLE" in rules(result)


def test_conflicting_historical_roles_on_same_photo(example):
    reg, pol, row, refs, record, check, _ = example
    train = replace(row, historical_role=Role.TRAIN)
    val = replace(
        record("image:val"),
        parents=(Derivation(row.id, Transform.CROP, refs),),
        historical_role=Role.VALIDATION,
    )
    policy = replace(
        pol,
        locks=(Lock(train.id, Role.TRAIN, refs), Lock(val.id, Role.VALIDATION, refs)),
    )
    result = check(replace(reg, records=(train, val)), policy)
    assert not result.valid and "CONTRADICTORY_HISTORICAL_ROLES" in rules(result)


def test_similarity_with_missing_endpoint_fails_closed(example):
    reg, pol, row, refs, _, check, _ = example
    pending = Candidate("candidate:missing", row.id, "image:missing", "pHash", refs)
    result = check(replace(reg, candidates=(pending,)), pol)
    assert not result.valid and "MISSING_CANDIDATE_ENDPOINT" in rules(result)

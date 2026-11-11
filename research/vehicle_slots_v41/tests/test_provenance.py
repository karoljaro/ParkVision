from dataclasses import replace

import pytest

from research.datasets_v4.contracts import (
    Asset,
    Assignment,
    Dataset,
    Derivation,
    Evidence,
    Fact,
    Knowledge,
    License,
    Lock,
    Manifest,
    Method,
    Policy,
    Record,
    Registry,
    Role,
    Transform,
    Verification,
)
from research.datasets_v4.storage import fingerprint
from research.vehicle_slots_v41.provenance import Provenance


@pytest.fixture
def registered(tmp_path):
    asset = Asset(str(tmp_path / "asset.jpg"), "0" * 64)
    ev = Evidence("ev:one", asset, "/frame", Method.MANIFEST, Verification.CONFIRMED)
    refs = (ev.id,)
    dataset = Dataset(
        "dataset:one",
        "revision",
        "test",
        License(
            "test",
            None,
            Verification.CONFIRMED,
            True,
            refs,
        ),
        refs,
    )
    source = Record(
        "source:frame",
        dataset.id,
        "frame",
        asset,
        Verification.CONFIRMED,
        site=Fact(Knowledge.KNOWN, "site:one", refs),
        camera=Fact(Knowledge.KNOWN, "camera:one", refs),
        historical_role=Role.TRAIN,
        evidence=refs,
    )
    return asset, ev, dataset, source


def test_protected_source_and_alias_never_enter_review(registered):
    asset, ev, dataset, source = registered
    source = replace(source, historical_role=Role.PROTECTED_TEST)
    alias = replace(
        source,
        id="source:alias",
        asset=replace(asset, sha256="1" * 64),
        parents=(Derivation(source.id, Transform.CROP, (ev.id,)),),
        historical_role=Role.QUARANTINE,
    )
    content = replace(source, id="source:same-pixels", historical_role=Role.TRAIN)
    same_site = replace(
        source,
        id="source:same-site",
        asset=replace(asset, sha256="2" * 64),
        historical_role=Role.DIAGNOSTIC,
    )
    records = (source, alias, content, same_site)
    registry = Registry("registry", (dataset,), records, (ev,))
    policy = Policy(
        "policy",
        (Lock(source.id, Role.PROTECTED_TEST, (ev.id,)),),
        ("site:one",),
        (),
        (asset.sha256,),
        (ev,),
    )
    manifest = Manifest(
        "curated",
        fingerprint(registry),
        fingerprint(policy),
        tuple(Assignment(r.id, r.historical_role) for r in records),
    )
    provenance = Provenance(registry, policy, manifest)
    for record in records:
        with pytest.raises(ValueError, match="protected v4 source"):
            provenance.review_source(record.id)


def test_unknown_camera_and_changed_registry_are_blocked(registered):
    _, ev, dataset, source = registered
    registry = Registry("registry", (dataset,), (source,), (ev,))
    policy = Policy("policy", (), (), (), (), (ev,))
    manifest = Manifest(
        "curated",
        fingerprint(registry),
        fingerprint(policy),
        (Assignment(source.id, Role.TRAIN),),
    )
    assert (
        Provenance(registry, policy, manifest).review_source(source.id)[1] == Role.TRAIN
    )
    changed = replace(registry, records=(replace(source, camera=Fact()),))
    with pytest.raises(ValueError, match="does not bind"):
        Provenance(changed, policy, manifest)
    matching = replace(manifest, registry_sha256=fingerprint(changed))
    with pytest.raises(ValueError, match="camera requires review"):
        Provenance(changed, policy, matching).review_source(source.id)

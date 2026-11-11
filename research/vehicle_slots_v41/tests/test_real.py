from collections import Counter
import os
from pathlib import Path

import pytest

from research.datasets_v4.contracts import Manifest, Role, ValidationResult
from research.datasets_v4.storage import AssetVerifier, fingerprint, load
from research.vehicle_slots_v41.relations import Link, Queue, Relation

LOCAL = os.environ.get("PARKVISION_V41_ARTIFACTS")


@pytest.mark.skipif(not LOCAL, reason="Set PARKVISION_V41_ARTIFACTS to prepared output")
def test_local_source_queue_uses_the_v4_gate():
    path = Path(LOCAL)
    queue = load(path / "queue.json", Queue)
    manifest = load(path / "diagnostic-v4-manifest.json", Manifest)
    gate = load(path / "diagnostic-v4-validation.json", ValidationResult)
    assert gate.valid and not gate.issues
    assert fingerprint(manifest) == queue.diagnostic_manifest_sha256
    assert gate.registry_sha256 == queue.registry_sha256
    assert gate.policy_sha256 == queue.policy_sha256
    assert not queue.training_eligible and len(queue.frames) == 5
    assert [d.record for d in gate.decisions] == [
        f.registry_record for f in queue.frames
    ]
    assert all(
        d.accepted and d.effective_role == Role.DIAGNOSTIC for d in gate.decisions
    )
    verifier = AssetVerifier()
    assert all(verifier.verify(b.archive) for b in queue.bundles)
    assert all(
        verifier.verify(f.image) and verifier.verify(f.source_subset)
        for f in queue.frames
    )
    assert all(f.annotation_rectangle == (0, 0, 1280, 720) for f in queue.frames)
    assert Counter(s.link for f in queue.frames for s in f.slots) == {
        Link.MATCHED: 102,
        Link.NO_VEHICLE: 80,
        Link.AMBIGUOUS: 1,
    }
    [ambiguous] = [s for f in queue.frames for s in f.slots if s.link == Link.AMBIGUOUS]
    assert ambiguous.car_id == 12000005869
    assert ambiguous.car_annotation_ids == (12000075936, 12000075953)
    [absent] = [f for f in queue.frames if not f.cars_frame_present]
    assert absent.image_id == 12000003661 and not absent.pairs
    assert sum(len(f.pairs) for f in queue.frames) == 417
    assert all(
        p.relation == Relation.NOT_ASSESSED for f in queue.frames for p in f.pairs
    )

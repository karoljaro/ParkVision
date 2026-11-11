from collections import Counter
from pathlib import Path
import os

import pytest

from research.datasets_v4.contracts import Manifest, Role, ValidationResult
from research.datasets_v4.storage import AssetVerifier, fingerprint, load
from research.vehicle_slots_v41.prepare import asset
from research.vehicle_slots_v41.relations import Queue, Relation

from research.vehicle_slots_v42.batch import known_pairs
from research.vehicle_slots_v42.contracts import Batch, Progress
from research.vehicle_slots_v42.export import reconcile

LOCAL = os.environ.get("PARKVISION_V42_ARTIFACTS")


@pytest.mark.skipif(not LOCAL, reason="Set PARKVISION_V42_ARTIFACTS to prepared output")
def test_real_twenty_pair_batch_with_preserved_gate_and_sources():
    root = Path(LOCAL)
    batch = load(root / "batch.json", Batch)
    gate = load(root / "diagnostic-v4-validation.json", ValidationResult)
    manifest = load(root / "diagnostic-v4-manifest.json", Manifest)
    verifier = AssetVerifier()
    assert verifier.verify(batch.queue)
    queue = load(Path(batch.queue.path), Queue)
    pairs = known_pairs(queue)
    assert len(batch.examples) == len({ex.pair for ex in batch.examples}) == 20
    assert Counter(e.suggestion.suggested_relation for e in batch.examples) == {
        Relation.OWNING: 7,
        Relation.PERSPECTIVE: 3,
        Relation.UNCERTAIN: 10,
    }
    assert len({e.pair.image_id for e in batch.examples}) == 3
    assert len({e.camera_id for e in batch.examples}) == 1
    assert not batch.training_eligible and gate.valid and not gate.issues
    assert fingerprint(manifest) == batch.diagnostic_manifest_sha256
    assert gate.registry_sha256 == batch.registry_sha256 == queue.registry_sha256
    assert gate.policy_sha256 == batch.policy_sha256 == queue.policy_sha256
    assert {
        d.record
        for d in gate.decisions
        if d.accepted and d.effective_role == Role.DIAGNOSTIC
    } == {e.registry_record for e in batch.examples}
    assert all(verifier.verify(bundle.archive) for bundle in batch.bundles)
    for ex in batch.examples:
        assert verifier.verify(ex.source) and verifier.verify(ex.source_subset)
        assert ex.capture_time.startswith("2012-12")
        assert pairs[ex.pair][1].relation == Relation.NOT_ASSESSED
        assert asset(root / "public" / ex.original_view).sha256 == ex.source.sha256
        for path in (ex.full_overlay, ex.detail_original, ex.detail_overlay):
            assert (root / "public" / path).is_file()
    assert reconcile(batch, Progress(fingerprint(batch), ()), queue) == ()

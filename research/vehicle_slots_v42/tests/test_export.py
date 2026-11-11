from dataclasses import replace

import pytest

from research.datasets_v4.contracts import Fact, Knowledge, Role
from research.datasets_v4.storage import fingerprint, load, save
from research.vehicle_slots_v41.prepare import asset
from research.vehicle_slots_v41.relations import (
    Frame,
    Link,
    Pair,
    Queue,
    Relation,
    Slot,
    Vehicle,
)
from research.vehicle_slots_v41.review import read_csv

from research.vehicle_slots_v42.contracts import (
    Batch,
    Confidence,
    Decision,
    Example,
    Geometry,
    PairKey,
    Progress,
    Status,
    Suggestion,
    Visibility,
)
from research.vehicle_slots_v42.export import export, reconcile


@pytest.fixture
def fixture(tmp_path):
    image = tmp_path / "source.jpg"
    image.write_bytes(b"original fixture image")
    subset = tmp_path / "subset.json"
    subset.write_text("{}")
    keys = PairKey(11, 21, 31), PairKey(11, 21, 32), PairKey(11, 21, 33)
    slots = tuple(
        Slot(
            k.spot_annotation_id,
            f"slot:{i}",
            99,
            1,
            Link.MATCHED,
            (21,),
            True,
            (),
        )
        for i, k in enumerate(keys)
    )
    site = Fact(Knowledge.KNOWN, "site:one", ("ev:site",))
    camera = Fact(Knowledge.KNOWN, "camera:one", ("ev:camera",))
    frame = Frame(
        11,
        asset(image),
        "source.jpg",
        1280,
        720,
        "record:one",
        Role.TRAIN,
        Role.TRAIN,
        Role.DIAGNOSTIC,
        site,
        camera,
        Fact(),
        Fact(),
        "2012-12-12T10:30:05",
        asset(subset),
        True,
        (0, 0, 1280, 720),
        (),
        slots,
        (Vehicle(21, 99, True, True, ()),),
        tuple(
            Pair(k.image_id, k.car_annotation_id, k.spot_annotation_id, "source")
            for k in keys
        ),
    )
    queue = Queue(*("a" * 64,) * 4, (), (frame,))
    source_queue = tmp_path / "queue.json"
    save(source_queue, queue)
    suggestion = Suggestion(
        Relation.OWNING,
        Geometry.PARTIAL,
        Visibility.PARTIAL,
        Confidence.MEDIUM,
        "Visible wheel",
        "Only one boundary visible",
    )
    examples = tuple(
        Example(
            f"AB{i:02}",
            k,
            99,
            s.camera_slot_key,
            site.value,
            camera.value,
            None,
            frame.source_capture_time,
            frame.registry_record,
            frame.image,
            frame.source_subset,
            frame.historical_role,
            (31,),
            (32,),
            "images/original.jpg",
            "images/full.png",
            "images/detail.png",
            "images/detail-overlay.png",
            suggestion,
        )
        for i, (k, s) in enumerate(zip(keys, slots, strict=True), start=1)
    )
    batch = Batch(
        asset(source_queue),
        fingerprint(queue),
        queue.registry_sha256,
        queue.policy_sha256,
        "b" * 64,
        (),
        examples,
    )
    return batch, queue, keys


def decision(key, status=Status.ACCEPTED, relation=Relation.OWNING):
    return Decision(
        key,
        status,
        relation,
        "Operator fixture",
        "2026-11-10T10:11:12+01:00",
        'Wheel near boundary, operator wrote "line".\nSecond observation.',
        Geometry.PARTIAL,
        Visibility.PARTIAL,
    )


def run_export(tmp_path, batch, decisions):
    batch_path = tmp_path / "batch.json"
    progress_path = tmp_path / "progress.json"
    output = tmp_path / "output"
    save(batch_path, batch)
    progress = Progress(fingerprint(batch), decisions)
    save(progress_path, progress)
    assert load(progress_path, Progress) == progress
    return output, export(batch_path, progress_path, output)


def test_suggestions_and_pending_decisions_are_not_accepted(tmp_path, fixture):
    batch, queue, keys = fixture
    output, count = run_export(
        tmp_path,
        batch,
        (
            decision(keys[0], Status.PENDING, None),
            decision(keys[1], Status.REJECTED, None),
            decision(keys[2], Status.NEEDS_REVIEW, None),
        ),
    )
    assert count == 0
    assert not read_csv(output / "v41-reviews.csv")
    assert load(output / "reviewed-v41-queue.json", Queue).frames == queue.frames


def test_v41_round_trip_preserves_multiple_slots_and_operator_provenance(
    tmp_path, fixture
):
    batch, queue, keys = fixture
    original_bytes = (tmp_path / "queue.json").read_bytes()
    output, count = run_export(
        tmp_path,
        batch,
        (
            decision(keys[0]),
            decision(keys[1], relation=Relation.BLOCKS),
            decision(keys[2], relation=Relation.UNCERTAIN),
        ),
    )
    assert count == 3
    reviews = read_csv(output / "v41-reviews.csv")
    assert {r.car_annotation_id for r in reviews} == {21}
    assert [r.relation for r in reviews] == [
        Relation.OWNING,
        Relation.BLOCKS,
        Relation.UNCERTAIN,
    ]
    assert all(
        r.reviewer == "Operator fixture" and "2026-11-10" in r.note for r in reviews
    )
    result = load(output / "reviewed-v41-queue.json", Queue)
    assert not result.training_eligible and result.parent_queue_sha256 == fingerprint(
        queue
    )
    assert result.frames[0].review_role == Role.DIAGNOSTIC
    assert (tmp_path / "queue.json").read_bytes() == original_bytes
    assert asset(tmp_path / "source.jpg") == batch.examples[0].source
    import json

    enriched = json.loads((output / "decisions.json").read_text())
    assert not enriched["training_eligible"]
    assert enriched["registry_sha256"] == batch.registry_sha256
    assert enriched["records"][0]["evidence"]["car_id"] == 99
    assert enriched["records"][0]["evidence"]["camera_slot_key"] == "slot:0"
    assert enriched["records"][0]["decision"]["reviewed_at"].endswith("+01:00")


def test_unknown_duplicate_and_wrong_batch_rejected(fixture):
    batch, queue, keys = fixture
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        reconcile(
            batch, Progress(fingerprint(batch), (decision(PairKey(11, 99, 31)),)), queue
        )
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        reconcile(batch, Progress(fingerprint(batch), (decision(keys[0]),) * 2), queue)
    with pytest.raises(ValueError, match="identities"):
        reconcile(batch, Progress("c" * 64, ()), queue)
    with pytest.raises(ValueError, match="identities"):
        reconcile(
            batch,
            Progress(fingerprint(batch), ()),
            replace(queue, training_eligible=True),
        )


def test_mismatched_provenance_and_changed_bytes_rejected(tmp_path, fixture):
    batch, queue, keys = fixture
    changed = replace(batch, examples=(replace(batch.examples[0], car_id=999),))
    with pytest.raises(ValueError, match="disagrees"):
        reconcile(changed, Progress(fingerprint(changed), (decision(keys[0]),)), queue)
    (tmp_path / "source.jpg").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="source queue, image or annotations changed"):
        run_export(tmp_path, batch, (decision(keys[0]),))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("status", "rel"),
    [
        (Status.ACCEPTED, None),
        (Status.ACCEPTED, Relation.NOT_ASSESSED),
        (Status.REJECTED, Relation.OWNING),
        (Status.NEEDS_REVIEW, Relation.BLOCKS),
    ],
)
def test_operator_must_explicitly_accept_valid_relation(status, rel):
    with pytest.raises(ValueError):
        decision(PairKey(11, 21, 31), status, rel)

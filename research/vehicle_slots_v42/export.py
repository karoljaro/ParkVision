"""Validate progress against the original batch and write a diagnostic v4.1 export."""

import csv
from dataclasses import dataclass
from pathlib import Path

from research.datasets_v4.storage import AssetVerifier, fingerprint, load, save
from research.vehicle_slots_v41.relations import Queue
from research.vehicle_slots_v41.review import Review, apply, read_csv
from research.vehicle_slots_v41.source import Bundle

from .batch import known_pairs
from .contracts import Batch, Decision, Example, Progress, Status


@dataclass(frozen=True, slots=True)
class RecordedDecision:
    evidence: Example
    decision: Decision


@dataclass(frozen=True, slots=True)
class ExportedDecisions:
    batch_sha256: str
    registry_sha256: str
    policy_sha256: str
    bundles: tuple[Bundle, ...]
    records: tuple[RecordedDecision, ...]
    training_eligible: bool = False


def reconcile(
    batch: Batch, progress: Progress, queue: Queue
) -> tuple[RecordedDecision, ...]:
    if (
        batch.training_eligible
        or queue.training_eligible
        or batch.schema_version != 1
        or progress.schema_version != 1
        or progress.batch_sha256 != fingerprint(batch)
        or batch.queue_sha256 != fingerprint(queue)
        or batch.registry_sha256 != queue.registry_sha256
        or batch.policy_sha256 != queue.policy_sha256
        or batch.bundles != queue.bundles
    ):
        raise ValueError("Progress, source queue and batch identities must agree")
    known = known_pairs(queue)
    examples = {e.pair: e for e in batch.examples}
    if len(examples) != len(batch.examples):
        raise ValueError("Duplicate pair in batch")
    result = []
    seen = set()
    for d in progress.decisions:
        if d.pair in seen or d.pair not in examples or d.pair not in known:
            raise ValueError(f"Unknown or duplicate decision pair: {d.pair.text}")
        seen.add(d.pair)
        example = examples[d.pair]
        frame, pair = known[d.pair]
        spot = next(
            s for s in frame.slots if s.annotation_id == pair.spot_annotation_id
        )
        car = next(
            c for c in frame.vehicles if c.annotation_id == pair.car_annotation_id
        )
        if (
            example.registry_record != frame.registry_record
            or example.source != frame.image
            or example.source_subset != frame.source_subset
            or example.camera_id != frame.camera.value
            or example.site_id != frame.site.value
            or example.car_id != car.car_id
            or example.camera_slot_key != spot.camera_slot_key
        ):
            raise ValueError(f"Pair evidence disagrees with v4.1: {d.pair.text}")
        result.append(RecordedDecision(example, d))
    return tuple(result)


def export(batch_path: Path, progress_path: Path, output: Path) -> int:
    batch = load(batch_path, Batch)
    progress = load(progress_path, Progress)
    verifier = AssetVerifier()
    assets = {batch.queue, *(bundle.archive for bundle in batch.bundles)}
    for example in batch.examples:
        assets.update((example.source, example.source_subset))
    if not all(verifier.verify(a) for a in assets):
        raise ValueError("The source queue, image or annotations changed")
    queue = load(Path(batch.queue.path), Queue)
    records = reconcile(batch, progress, queue)
    accepted = [
        r.decision for r in records if r.decision.review_status == Status.ACCEPTED
    ]
    reviews = tuple(
        Review(
            d.pair.image_id,
            d.pair.car_annotation_id,
            d.pair.spot_annotation_id,
            d.operator_relation,
            d.reviewer,
            f"[{d.reviewed_at}] {d.note}",
        )
        for d in accepted
    )
    reviewed = apply(queue, reviews)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "v41-reviews.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "image_id",
                "car_annotation_id",
                "spot_annotation_id",
                "relation",
                "reviewer",
                "note",
            )
        )
        for row in reviews:
            writer.writerow(
                (
                    row.image_id,
                    row.car_annotation_id,
                    row.spot_annotation_id,
                    row.relation,
                    row.reviewer,
                    row.note,
                )
            )
    if read_csv(csv_path) != reviews:
        raise ValueError("v4.1 CSV did not round trip")
    save(output / "reviewed-v41-queue.json", reviewed)
    save(
        output / "decisions.json",
        ExportedDecisions(
            fingerprint(batch),
            batch.registry_sha256,
            batch.policy_sha256,
            batch.bundles,
            records,
        ),
    )
    return len(accepted)

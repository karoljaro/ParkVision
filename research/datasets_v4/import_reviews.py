"""Only reviewed audit findings become graph edges. Hash/ORB hits stay queued."""

import json
from pathlib import Path

from .contracts import (
    Candidate,
    Derivation,
    Method,
    RelationKind,
    ReviewState,
    Transform,
)
from .import_context import ImportContext, lines
from .storage import encode


def import_confirmed(ctx: ImportContext) -> set[frozenset[str]]:
    confirmed_pairs: set[frozenset[str]] = set()
    review_path = ctx.audit / "technical/visual-review.json"
    for row in ctx.document(review_path)["cross_dataset_pairs"]:
        if row["visual_review"]["reviewed"] is not True:
            raise ValueError("Unreviewed relation cannot be imported as confirmed")
        left, right = ctx.technical_ids[row["i"]], ctx.technical_ids[row["j"]]
        ev = ctx.ev(
            review_path,
            f"/cross_dataset_pairs: i={row['i']},j={row['j']}",
            Method.VISUAL,
        )
        ctx.link(left, right, RelationKind.SAME_PHOTOGRAPH, ev)
        confirmed_pairs.add(frozenset((left, right)))
    proof = ctx.audit / "leakage/previous-source-overlap-evidence.jpg"
    for number, source_name, transform in (
        (474, "GOPR6536.JPG", Transform.ROTATION),
        (515, "GOPR6721.JPG", Transform.REENCODE),
    ):
        parent = ctx.by_source["ACPDS/" + source_name]
        child = next(
            identity
            for identity, meta in ctx.original.items()
            if meta["archive"] == "parking spaces.v1i.coco-segmentation.zip"
            and meta["split"] == "train"
            and meta["image_id"] == number
        )
        ev = ctx.ev(
            proof, source_name + ": visually confirmed source frame", Method.VISUAL
        )
        row = ctx.records[child]
        ctx.patch(
            child,
            parents=(
                *row.parents,
                Derivation(
                    parent, transform, (ev,), '{"exact_transform_parameters":null}'
                ),
            ),
        )
        confirmed_pairs.add(frozenset((child, parent)))
    # This is a different day at the same parking, NOT a same-photo assertion.
    child = next(
        identity
        for identity, meta in ctx.original.items()
        if meta["archive"] == "parking spaces.v1i.coco-segmentation.zip"
        and meta["split"] == "train"
        and meta["image_id"] == 1353
    )
    ev = ctx.ev(proof, "PUCPR: same physical parking; different date", Method.VISUAL)
    source_ev = ctx.ev(
        ctx.audit / "leakage/source-name-overlaps-and-protection.json",
        "metapklot_named_matches: source day 2012-11-06",
        Method.MANIFEST,
    )
    ctx.patch(
        child,
        site=ctx.fact("site:pklot-pucpr", ev),
        day=ctx.fact("2012-11-06", source_ev),
    )
    protected = next(
        row.id
        for row in ctx.records.values()
        if row.id.startswith("native:") and row.site.value == "site:pklot-pucpr"
    )
    ctx.link(child, protected, RelationKind.LOCATION_DEPENDENCY, ev)
    road_path = ctx.audit / "leakage/road-same-site-adjudication.json"
    for index, pair in enumerate(ctx.document(road_path)["confirmed_pairs"]):
        left = "raw:" + pair["train"]["id"]
        right = "raw:" + pair["val"]["id"]
        ev = ctx.ev(
            road_path,
            f"/confirmed_pairs/{index}; static background reviewed",
            Method.VISUAL,
        )
        ctx.link(left, right, RelationKind.LOCATION_DEPENDENCY, ev)
        confirmed_pairs.add(frozenset((left, right)))
    for index, pair in enumerate(ctx.document(review_path)["road"]):
        if pair["status"] == "vehicle_identity_only":
            ev = ctx.ev(
                review_path,
                f"/road/{index}; matches a vehicle, not static site",
                Method.VISUAL,
            )
            candidate = Candidate(
                f"reviewed:orb:{index}",
                "raw:" + pair["train_id"],
                "raw:" + pair["val_id"],
                "ORB site hypothesis rejected; origin still unknown",
                (ev,),
                ReviewState.REJECTED,
            )
            ctx.reviewed[candidate.id] = candidate
    return confirmed_pairs


def export_candidate_queue(ctx: ImportContext, confirmed_pairs: set[frozenset[str]]):
    path = ctx.output / "review-candidates.jsonl"
    near = ctx.audit / "technical/near-pairs.jsonl"
    ev = ctx.ev(near, "/; candidate retrieval is not source identity", Method.HEURISTIC)
    count = 0
    with path.open("x") as stream:

        def emit(left: str, right: str, method: str, evidence: str) -> None:
            nonlocal count
            if left == right or frozenset((left, right)) in confirmed_pairs:
                return
            candidate = Candidate(
                f"candidate:{count}", left, right, method, (evidence,)
            )
            stream.write(json.dumps(encode(candidate), sort_keys=True) + "\n")
            count += 1

        for pair in lines(near):
            # Exact byte/pixel content is independently checked by the gate;
            # it does not claim that black tiles share a physical source.
            if pair["exact_pixel"]:
                continue
            emit(
                ctx.technical_ids[pair["i"]],
                ctx.technical_ids[pair["j"]],
                f"pHash={pair['phash_distance']};dHash={pair['dhash_distance']};"
                f"NCC64={pair['gray_ncc64']}",
                ev,
            )
        external = list(lines(ctx.audit / "technical/external-images.jsonl"))
        extra = ctx.audit / "technical/external-near-pairs.jsonl"
        ext_ev = ctx.ev(extra, "/; no automatic source linkage", Method.HEURISTIC)
        for pair in lines(extra):
            target_path = str(Path(external[pair["external"]]["path"]).resolve())
            if target_path not in ctx.by_path:
                raise ValueError(
                    f"External candidate record not imported: {target_path}"
                )
            emit(
                ctx.technical_ids[pair["i"]],
                ctx.by_path[target_path],
                f"external-pHash={pair['phash_distance']};"
                f"dHash={pair['dhash_distance']}",
                ext_ev,
            )
        names_path = ctx.audit / "leakage/source-name-overlaps-and-protection.json"
        names_ev = ctx.ev(
            names_path, "/gopr_matches; filename-only candidates", Method.HEURISTIC
        )
        for row in ctx.document(names_path)["gopr_matches"]:
            left = ctx.raw[(row["archive"], row["member"])]
            parent = ctx.by_source["ACPDS/" + row["gopr"] + ".JPG"]
            emit(
                left,
                parent,
                "retained filename; requires visual/source confirmation",
                names_ev,
            )
        orb_path = ctx.audit / "technical/road-geometric.json"
        orb_ev = ctx.ev(
            orb_path,
            "/pairs; ORB/RANSAC retrieval alone is not proof of site",
            Method.HEURISTIC,
        )
        rejected = {frozenset((c.left, c.right)) for c in ctx.reviewed.values()}
        for pair in ctx.document(orb_path)["pairs"]:
            left, right = "raw:" + pair["train_id"], "raw:" + pair["val_id"]
            if frozenset((left, right)) not in rejected:
                emit(left, right, f"ORB:RANSAC_inliers={pair['inliers']}", orb_ev)
    return ctx.asset(path), count

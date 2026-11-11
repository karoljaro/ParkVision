"""Import existing corrected human labels and 24 deterministic local queue items."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from research.occupancy.training_v1.data import file_hash, local_path, write_json
from research.occupancy.visibility_v2.dataset import build, historical_visibility

CNR_TERMS = (
    "ODbL-1.0 per MetaPKLot; CNRPark-EXT Amato et al.; "
    "upstream terms not independently confirmed; local only"
)


def reference(path):
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def record(root, frame, index, site, camera, license_info):
    path = local_path(root, frame["image"])
    if "rois" in frame:
        roi = frame["rois"][index]
    else:
        roi = (
            np.float32(frame["slots"][index]["roi"])
            / np.float32((frame["width"] - 1, frame["height"] - 1))
        ).tolist()
    source_path = frame.get("source_path", "ACPDS/test/" + Path(frame["image"]).name)
    geometry = sha256(json.dumps(roi, separators=(",", ":")).encode()).hexdigest()[:16]
    day = frame.get("day", frame.get("date"))
    if isinstance(day, list):
        day = "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(day))
    return {
        "image_path": str(path),
        "image_sha256": frame["sha256"],
        "source_path": source_path,
        "roi_index": index,
        "roi_normalized": roi,
        "site_id": site,
        "camera_id": camera,
        "day_id": day,
        "slot_projection_id": f"{camera or source_path}:{geometry}",
        "physical_slot_id": None,
        "region_role": "physical_parking_slot",
        "license": license_info,
        "visibility": "unassessed",
        "references": [],
        "source_state_is_visibility": False,
    }


def prepare(workspace, datasets, output):
    output.mkdir(parents=True, exist_ok=False)
    base = workspace / "occupancy-pilot"
    # Latest corrections win by explicit source selection, never by mtime.
    specs = [
        (
            "paired-acpds-2026-11-03-v1",
            "visibility-review/queue.json",
            "run-v1/manual-correction-01/acpds-visibility-labels.json",
            None,
            None,
            "MIT; ACPDS Martin Marek",
        ),
        (
            "fixed-cnr6-2026-11-03-v1",
            "visibility-review/queue.json",
            "run-v1/manual-correction-01/cnr6-visibility-labels.json",
            "pisa-cnr",
            "camera6",
            CNR_TERMS,
        ),
        (
            "fixed-ufpr05-conflict-2026-11-03-v1",
            "visibility-review/queue.json",
            "run-v1/ufpr05-conflict-labels.json",
            "ufpr",
            "UFPR05",
            "CC-BY-4.0; PKLot Almeida et al.",
        ),
        (
            "fixed-ufpr05-conflict-2026-11-03-v1",
            "conflict-review/queue.json",
            "run-v1/ufpr05-conflict-followup-labels.json",
            "ufpr",
            "UFPR05",
            "CC-BY-4.0; PKLot Almeida et al.",
        ),
        (
            "fixed-ufpr05-conflict-2026-11-03-v1",
            "residual-free-review/queue.json",
            "run-v1/ufpr05-residual-free-labels.json",
            "ufpr",
            "UFPR05",
            "CC-BY-4.0; PKLot Almeida et al.",
        ),
        (
            "weak-free-veto-2026-11-03-v1",
            "weak-veto-review/queue.json",
            "ufpr05-weak-veto-labels.json",
            "ufpr",
            "UFPR05",
            "CC-BY-4.0; PKLot Almeida et al.",
        ),
        (
            "projection-constraints-2026-11-03-v1/cnr6-trial-v1",
            "visibility-review/queue.json",
            "run-v1/cnr6-projection-labels.json",
            "pisa-cnr",
            "camera6",
            CNR_TERMS,
        ),
    ]
    records, audits = [], []
    for directory, queue_relative, labels_relative, site, camera, license_info in specs:
        root = base / directory
        manifest_path, queue_path, labels_path = (
            root / "manifest.json",
            root / queue_relative,
            root / labels_relative,
        )
        manifest, queue, labels = [
            json.loads(p.read_text()) for p in (manifest_path, queue_path, labels_path)
        ]
        if (
            queue["source_manifest_sha256"] != file_hash(manifest_path)
            or labels["queue_sha256"] != file_hash(queue_path)
            or set(labels["records"]) != {r["id"] for r in queue["observations"]}
        ):
            raise ValueError("Historical label/queue/manifest mismatch")
        frames = {f["image"]: f for f in manifest["frames"]}
        for item in queue["observations"]:
            label = labels["records"][item["id"]]
            r = record(
                root,
                frames[item["image"]],
                item["roi_index"],
                site,
                camera,
                license_info,
            )
            r.update(
                visibility=historical_visibility(label),
                historical_label=label,
                manual_id=item["id"],
                selection="historical diagnostic queue; not random",
                references=[
                    reference(p) for p in (manifest_path, queue_path, labels_path)
                ],
            )
            records.append(r)
        audits.append(
            {
                "labels": reference(labels_path),
                "n": len(queue["observations"]),
                "classes": dict(
                    Counter(
                        historical_visibility(v) for v in labels["records"].values()
                    )
                ),
            }
        )
    # One new ROI per day, selected by identity hash and never by model confidence.
    # These remain unassessed. Reuse existing images; no downloads or test paint.
    for site, manifest_path, allowed_camera in (
        (
            "ufpr",
            workspace
            / "dataset-model-improvement-2026-11-08-v1"
            / "expansion-multisite-v1/data/manifest.json",
            "UFPR04",
        ),
        ("pisa-cnr", base / "controlled-eval-2026-11-03-v1/manifest.json", "camera3"),
    ):
        manifest = json.loads(manifest_path.read_text())
        frames = [f for f in manifest["frames"] if f["camera_id"] == allowed_camera]
        frames.sort(
            key=lambda f: sha256(
                ("visibility-v2:" + f["source_path"]).encode()
            ).hexdigest()
        )
        days, count = set(), 0
        for frame in frames:
            day = frame.get("day", frame.get("date"))
            day = tuple(day) if isinstance(day, list) else day
            if day in days:
                continue
            if "slots" in frame:
                index = int(
                    sha256(frame["source_path"].encode()).hexdigest()[:8], 16
                ) % len(frame["slots"])
                r = record(
                    manifest_path.parent,
                    frame,
                    index,
                    site,
                    allowed_camera,
                    CNR_TERMS,
                )
            else:
                samples = [
                    s
                    for s in manifest["samples"]
                    if s["source_path"] == frame["source_path"]
                ]
                samples.sort(
                    key=lambda s: sha256(
                        ("visibility-v2:" + s["id"]).encode()
                    ).hexdigest()
                )
                sample = samples[0]
                normalized = (
                    np.float32(sample["roi"])
                    / np.float32((frame["width"] - 1, frame["height"] - 1))
                ).tolist()
                r = record(
                    manifest_path.parent,
                    dict(frame, rois=[normalized]),
                    0,
                    site,
                    allowed_camera,
                    "CC-BY-4.0; PKLot Almeida et al.",
                )
                r["source_annotation_id"] = sample["source_annotation_id"]
                r["slot_projection_id"] = sample["slot_projection_id"]
            r.update(
                selection="identity hash; one ROI/day; no model scores",
                references=[reference(manifest_path)],
            )
            records.append(r)
            count += 1
            days.add(day)
            if count == 12:
                break
        audits.append({"new_queue_site": site, "camera": allowed_camera, "n": count})
    archives = []
    for path in sorted(datasets.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            annotations = [n for n in archive.namelist() if n.endswith(".coco.json")]
            metadata = [json.loads(archive.read(n)) for n in annotations]
            archives.append(
                {
                    **reference(path),
                    "annotations": annotations,
                    "categories": metadata[0]["categories"] if metadata else [],
                    "exporter_license_declaration": (
                        metadata[0].get("licenses", []) if metadata else []
                    ),
                    "visibility_labels": 0,
                    "used": False,
                    "reason": (
                        "No visibility labels; unresolved site/camera/day and "
                        "individual rights. Export splits not trusted."
                    ),
                }
            )
    index = {
        "schema_version": 1,
        "site_splits": {"ufpr": "train", "pisa-cnr": "val"},
        "records": records,
    }
    index_path = output / "source-index.json"
    write_json(index_path, index)
    temporal_path = (
        workspace
        / "temporal-occupancy-v1-2026-11-05/clearpix-pilot-v1"
        / "evaluation-v1/labels-corrected.json"
    )
    temporal = json.loads(temporal_path.read_text())
    write_json(
        output / "source-audit.json",
        {
            "manual_sources": audits,
            "roboflow": archives,
            "clearpix": {
                **reference(temporal_path),
                "intervals": len(temporal["labels"]),
                "visibility": dict(
                    Counter(r["visibility"] for r in temporal["labels"])
                ),
                "region_6_parking_function": temporal["region_6_parking_function"],
                "used": False,
            },
            "excluded": [
                "Superseded original ACPDS/CNR labels; "
                "corrected revisions selected explicitly.",
                "Manual occupancy-only queue: no explicit visibility labels.",
                "ClearPix: 14 correlated temporal intervals (7 clear, 7 partial), "
                "no unreadable; crop/time alignment and rights need review, "
                "region 6 remains unconfirmed. "
                "Not converted to independent frame labels.",
                "PUCPR frozen paint test and T-01: never imported.",
            ],
        },
    )
    return build(index_path, output / "dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.datasets, args.output)["summary"]))


if __name__ == "__main__":
    main()

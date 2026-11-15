"""Same-camera manifest input and source-aware review of fused geometry."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from .capture.frame import Frame
from .detection.multi_frame import (
    GeometryFrame,
    MultiFrameInitialization,
    analyze_frames,
)
from .domain.types import CameraId, ParkingId
from .geometry_setup import _export_geometry, _png_bytes, _read_reference


def _sequence_preview(images: list, result: MultiFrameInitialization) -> np.ndarray:
    """Show per-frame axes/corners, fusion and rejected evidence at readable size."""
    from .geometry_setup_preview import render_initialization_preview

    width, height = 720, 470
    panels = []
    rejected = {
        item["observation_id"] for item in result.conflicts if "observation_id" in item
    }
    rejected_lines = {
        line_id for item in result.conflicts for line_id in item.get("line_ids", ())
    }
    rejected.update(
        o["observation_id"]
        for o in result.observations
        if o.get("fusion_line_id") in rejected_lines
    )
    for index, (reference, record) in enumerate(zip(images, result.frames)):
        image = reference.image.copy()
        for observation in result.observations:
            if observation["frame_index"] != index:
                continue
            line = observation["source_axis"]
            cv2.line(
                image,
                tuple(round(v) for v in line["start"]),
                tuple(round(v) for v in line["end"]),
                (
                    (30, 30, 230)
                    if observation["observation_id"] in rejected
                    else (0, 220, 255)
                ),
                2,
            )
        for node in record.get("source_topology", {}).get("junctions", []):
            cv2.circle(
                image, tuple(round(v) for v in node["position"]), 4, (255, 200, 0), 2
            )
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        factor = min(width / reference.width, (height - 70) / reference.height)
        resized = cv2.resize(
            image, (round(reference.width * factor), round(reference.height * factor))
        )
        panel[70 : 70 + resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(
            panel,
            f"F{index} {reference.path.name}"[:80],
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
        )
        cv2.putText(
            panel,
            record["reason"],
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 100, 20) if record["accepted"] else (30, 30, 200),
            1,
        )
        panels.append(panel)
    canvas = images[0].image.copy()
    line_map = {line.segment_id: line for line in result.geometry.topology.lines}
    for marking in result.markings:
        line = line_map.get(marking["line_id"])
        if line is None:
            continue
        multiple = marking["support_kind"] == "multiple_frames"
        start, end = [
            tuple(round(v) for v in point) for point in (line.start, line.end)
        ]
        cv2.line(
            canvas,
            start,
            end,
            (0, 220, 0) if multiple else (0, 180, 255),
            3 if multiple else 1,
        )
        cv2.putText(
            canvas,
            "/".join(f"F{i}" for i in marking["frame_indices"]),
            start,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    factor = min(width / canvas.shape[1], (height - 70) / canvas.shape[0])
    small = cv2.resize(
        canvas, (round(canvas.shape[1] * factor), round(canvas.shape[0] * factor))
    )
    panel[70 : 70 + small.shape[0], : small.shape[1]] = small
    cv2.putText(
        panel,
        "FUSED: green=multi-frame, gold=one frame",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
    )
    cv2.putText(
        panel,
        f"{len(result.conflicts)} conflicts quarantined; inspect evidence.json",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
    )
    panels.append(panel)
    while len(panels) % 2:
        panels.append(np.full_like(panels[0], 245))
    overview = np.vstack(
        [np.hstack(panels[i : i + 2]) for i in range(0, len(panels), 2)]
    )
    review = render_initialization_preview(images[0].image, result.geometry)
    review = cv2.resize(
        review,
        (
            overview.shape[1],
            round(review.shape[0] * overview.shape[1] / review.shape[1]),
        ),
    )
    return np.vstack((overview, review))


def propose_sequence(
    manifest_path: Path,
    output_dir: Path,
    *,
    parking_id: ParkingId,
    parking_name: str,
    camera_id: CameraId,
    camera_name: str,
    revision: int = 1,
) -> tuple[int, int]:
    """Manifest paths are relative to its file; first frame is the reference.

    Each frame MUST declare its own source and camera IDs. Shared defaults are
    deliberately disallowed because they hide mixed-camera input mistakes.
    """
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("frames"), list)
        or not manifest["frames"]
    ):
        raise ValueError("Manifest requires a nonempty frames array")
    if len(manifest["frames"]) > 32:
        raise ValueError("Select at most 32 diagnostic frames per review")
    images, frames = [], []
    for index, entry in enumerate(manifest["frames"]):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in ("image", "camera_id", "source_id")
        ):
            raise ValueError("Each frame requires image, camera_id and source_id")
        reference = _read_reference(manifest_path.parent / entry["image"])
        images.append(reference)
        frames.append(
            GeometryFrame(
                Frame(entry["camera_id"], index, datetime.now(UTC), reference.image),
                entry["source_id"],
                f"{index}:{reference.path}",
                reference.digest,
            )
        )
    if frames[0].frame.camera_id != camera_id:
        raise ValueError("Reference camera_id does not match requested camera")
    mask = None
    mask_path = manifest.get("background_mask")
    if mask_path:
        mask = cv2.imread(str(manifest_path.parent / mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Cannot read stationary background mask")
    result = analyze_frames(tuple(frames), background_mask=mask)
    metadata = {
        "schema_version": 1,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "data_role": manifest.get("role", "unverified_diagnostic"),
        "background_mask_pixel_sha256": (
            sha256(mask.tobytes()).hexdigest() if mask is not None else None
        ),
        "background_limit": "operator_selected_static_scenery_not_automatic_semantics",
        "frames": list(result.frames),
        "observations": list(result.observations),
        "markings": list(result.markings),
        "conflicts": list(result.conflicts),
        "hypothesis_updates": list(result.hypothesis_updates),
        "limits": [
            "missing_detection_is_unknown",
            "no_automatic_operator_verification",
            "finite_source_support_required",
            "no_confidence_boost_from_duplicates",
        ],
    }
    # Candidate lineage includes context, separators and retained reference ROI.
    for candidate_id, candidate in enumerate(result.geometry.candidates, 1):
        ids = set(candidate.separator_ids) | {
            i for b in candidate.boundaries for i in b.line_ids
        }
        metadata.setdefault("candidate_sources", []).append(
            {
                "candidate_id": candidate_id,
                "observation_ids": sorted(
                    o["observation_id"]
                    for o in result.observations
                    if o.get("fusion_line_id") in ids
                ),
                "missing_boundaries": candidate.missing_boundaries,
            }
        )
    return _export_geometry(
        images[0],
        result.geometry,
        output_dir,
        parking_id=parking_id,
        parking_name=parking_name,
        camera_id=camera_id,
        camera_name=camera_name,
        revision=revision,
        multi_frame=metadata,
        extra_files={
            "sequence-preview.png": _png_bytes(_sequence_preview(images, result))
        },
    )

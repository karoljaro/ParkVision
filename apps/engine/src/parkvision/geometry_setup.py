"""Prepare and inspect parking geometry for the existing LabelMe approval workflow."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from os.path import relpath
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .capture.frame import Frame
from .configuration import _integer, _string, load_configuration
from .configuration_preview import render_configuration_preview
from .detection.opencv_slot_detector import OpenCVSlotDetector
from .detection.initialization import GeometryInitialization
from .domain.geometry.polygon import Polygon
from .domain.types import CameraId, ParkingId
from .geometry_setup_preview import render_initialization_preview


@dataclass(frozen=True, slots=True)
class _Reference:
    path: Path
    image: NDArray[np.uint8]
    digest: str
    width: int
    height: int


def _read_reference(image_path: Path) -> _Reference:
    path = image_path.resolve(strict=True)
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Cannot decode reference image: {image_path}")
    encoded = np.frombuffer(raw, dtype=np.uint8)
    pixels = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if pixels is None:
        raise ValueError(f"Cannot decode reference image: {image_path}")
    height, width = pixels.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("Reference dimensions must both be >= 2 pixels")
    if not np.array_equal(pixels, cv2.imdecode(encoded, cv2.IMREAD_COLOR)):
        raise ValueError("Normalize the reference image's EXIF orientation first")
    return _Reference(
        path, cast(NDArray[np.uint8], pixels), sha256(raw).hexdigest(), width, height
    )


def _polygon_pixels(polygon: Polygon, reference: _Reference) -> list[list[float]]:
    return [
        [point.x * (reference.width - 1), point.y * (reference.height - 1)]
        for point in polygon.vertices
    ]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError("Cannot encode preview as PNG")
    return encoded.tobytes()


def _save_review(output_dir: Path, files: Mapping[str, bytes]) -> None:
    """Publish a fresh review directory without replacing an earlier review."""
    output_dir.mkdir()
    created: list[Path] = []
    try:
        for name, data in files.items():
            path = output_dir / name
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            # Leave unrelated files that appeared in the newly created directory.
            pass
        raise


def propose_geometry(
    image_path: Path,
    output_dir: Path,
    *,
    parking_id: ParkingId,
    parking_name: str,
    camera_id: CameraId,
    camera_name: str,
    revision: int = 1,
) -> tuple[int, int]:
    """Export an editable draft, diagnostic evidence, and a visual comparison."""
    reference = _read_reference(image_path)
    frame = Frame(camera_id, 0, datetime.now(UTC), reference.image)
    result = OpenCVSlotDetector().analyze(frame)
    return _export_geometry(
        reference,
        result,
        output_dir,
        parking_id=parking_id,
        parking_name=parking_name,
        camera_id=camera_id,
        camera_name=camera_name,
        revision=revision,
    )


def _export_geometry(
    reference: _Reference,
    result: GeometryInitialization,
    output_dir: Path,
    *,
    parking_id: ParkingId,
    parking_name: str,
    camera_id: CameraId,
    camera_name: str,
    revision: int = 1,
    multi_frame: dict | None = None,
    extra_files: Mapping[str, bytes] | None = None,
) -> tuple[int, int]:
    metadata = {
        "task": "parking-slot-setup",
        "parking_id": _string(parking_id, "parking_id"),
        "parking_name": _string(parking_name, "parking_name"),
        "camera_id": _string(camera_id, "camera_id"),
        "camera_name": _string(camera_name, "camera_name"),
        "revision": _integer(revision, "revision", minimum=1),
    }
    metadata["image_sha256"] = reference.digest
    relative_image = relpath(reference.path, output_dir.resolve())
    shapes: dict[str, dict[str, object]] = {}
    records = []
    for candidate_id, candidate in enumerate(result.candidates, start=1):
        slot_id = None
        pixels = None
        if candidate.proposal is not None:
            pixels = _polygon_pixels(candidate.proposal.polygon, reference)
            # Stable for repeated analysis of the same reference and geometry.
            # The editable label becomes the physical slot ID at review time.
            identity = [parking_id, camera_id, reference.digest, sorted(pixels)]
            digest = sha256(_json_bytes(identity)).hexdigest()[:20]
            slot_id = f"slot-{digest}"
            shapes.setdefault(
                slot_id,
                {
                    "label": slot_id,
                    "points": pixels,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                },
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "status": candidate.status,
                "slot_id": slot_id,
                "polygon": pixels,
                "boundaries": [asdict(boundary) for boundary in candidate.boundaries],
                "junction_ids": candidate.junction_ids,
                "reason": candidate.reason,
                "separator_ids": candidate.separator_ids,
                "rail_id": candidate.rail_id,
                "anchor_positions": candidate.anchor_positions,
                "geometry_kind": candidate.geometry_kind,
                "missing_boundaries": candidate.missing_boundaries,
                "row_id": candidate.row_id,
                "rejection_code": candidate.rejection_code,
                "rejection_line_ids": candidate.rejection_line_ids,
            }
        )

    draft = _json_bytes(
        {
            "version": "5.9.1",
            "flags": {},
            "shapes": list(shapes.values()),
            "imagePath": relative_image,
            "imageData": None,
            "imageHeight": reference.height,
            "imageWidth": reference.width,
            "parkvision": metadata,
        }
    )
    evidence = {
        "schema_version": 1,
        "initial_draft_sha256": sha256(draft).hexdigest(),
        "coordinate_space": "reference_image_pixel_centers",
        "reference": relative_image,
        "image_sha256": reference.digest,
        "image_width": reference.width,
        "image_height": reference.height,
        "camera_id": camera_id,
        "segments": [asdict(line) for line in result.segments],
        "fitted_lines": [asdict(line) for line in result.fitted_lines],
        "source_groups": [
            {
                "fitted_line_id": line_id,
                "segment_ids": [segment.segment_id for segment in group.segments],
            }
            for line_id, group in result.source_groups
        ],
        "marking_pairs": [asdict(pair) for pair in result.pairs],
        "topology": asdict(result.topology),
        "topology_support": [asdict(item) for item in result.support],
        "rows": [asdict(row) for row in result.rows],
        "candidates": records,
        "summary": {
            "full_roi": len(shapes),
            "partial_hypotheses": sum(
                candidate.geometry_kind == "partial"
                and candidate.rejection_code is None
                for candidate in result.candidates
            ),
            "rejected_partial_hypotheses": sum(
                candidate.geometry_kind == "partial"
                and candidate.rejection_code is not None
                for candidate in result.candidates
            ),
            "other_unresolved": sum(
                candidate.status == "unresolved"
                and candidate.geometry_kind != "partial"
                for candidate in result.candidates
            ),
            "open_rows": sum(
                row.geometry_kind in ("open_ended", "partial")
                and row.rejection_code is None
                for row in result.rows
            ),
        },
    }
    preview = render_initialization_preview(reference.image, result)
    if multi_frame is not None:
        evidence["multi_frame"] = multi_frame
    _save_review(
        output_dir,
        {
            "draft.json": draft,
            "evidence.json": _json_bytes(evidence),
            "preview.png": _png_bytes(preview),
            **(extra_files or {}),
        },
    )
    return len(shapes), sum(
        candidate.status == "unresolved" for candidate in result.candidates
    )


def export_verified_review(
    image_path: Path,
    configuration_path: Path,
    output_dir: Path,
    *,
    camera_id: CameraId | None = None,
) -> int:
    """Show operator-verified projections from an approved configuration revision."""
    configuration = load_configuration(configuration_path)
    if camera_id is None:
        if len(configuration.cameras) != 1:
            raise ValueError(
                "Select --camera-id for a configuration with several cameras"
            )
        camera_id = configuration.cameras[0].id
    reference = _read_reference(image_path)
    projections = configuration.projections_for_camera(
        camera_id, image_width=reference.width, image_height=reference.height
    )
    assert configuration.approved_at is not None  # Guaranteed by load_configuration.
    verified = {
        "schema_version": 1,
        "coordinate_space": "reference_image_pixel_centers",
        "parking_id": configuration.parking.id,
        "camera_id": camera_id,
        "revision": configuration.revision,
        "approved_by": configuration.approved_by,
        "approved_at": configuration.approved_at.isoformat(),
        "preview_image_sha256": reference.digest,
        "image_width": reference.width,
        "image_height": reference.height,
        "projections": [
            {
                "slot_id": projection.parking_slot_id,
                "status": "operator_verified",
                "polygon": _polygon_pixels(projection.polygon, reference),
            }
            for projection in projections
        ],
    }
    preview = render_configuration_preview(configuration, camera_id, reference.image)
    _save_review(
        output_dir,
        {"verified.json": _json_bytes(verified), "preview.png": _png_bytes(preview)},
    )
    return len(projections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser(
        "propose", help="Detect geometry and write a LabelMe draft"
    )
    verified = commands.add_parser("verified", help="Preview an approved configuration")
    sequence = commands.add_parser(
        "propose-sequence",
        help="Fuse a declared same-camera sequence into a LabelMe draft",
    )
    for command in (propose, verified):
        command.add_argument("--image", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    sequence.add_argument("--manifest", type=Path, required=True)
    sequence.add_argument("--output-dir", type=Path, required=True)
    for command in (propose, sequence):
        command.add_argument("--parking-id", type=ParkingId, required=True)
        command.add_argument("--parking-name", required=True)
        command.add_argument("--camera-id", type=CameraId, required=True)
        command.add_argument("--camera-name", required=True)
        command.add_argument("--revision", type=int, default=1)
    verified.add_argument("--camera-id", type=CameraId)
    verified.add_argument("--configuration", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("propose", "propose-sequence"):
            if args.command == "propose-sequence":
                from .geometry_sequence_setup import propose_sequence

                propose_function = propose_sequence
                input_path = args.manifest
            else:
                propose_function = propose_geometry
                input_path = args.image
            found, unresolved = propose_function(
                input_path,
                args.output_dir,
                parking_id=args.parking_id,
                parking_name=args.parking_name,
                camera_id=args.camera_id,
                camera_name=args.camera_name,
                revision=args.revision,
            )
            print(
                f"DRAFT: {found} proposed polygons, {unresolved} unresolved candidates"
            )
            evidence = json.loads((args.output_dir / "evidence.json").read_bytes())
            summary = evidence["summary"]
            print(
                f"PARTIAL: {summary['partial_hypotheses']} marking hypotheses, "
                f"{summary['open_rows']} open row structures"
            )
            print(
                f"REJECTED: {summary['rejected_partial_hypotheses']} "
                "non-parking partial hypotheses"
            )
            print(f"Open in LabelMe with --nodata: {args.output_dir / 'draft.json'}")
        else:
            count = export_verified_review(
                args.image,
                args.configuration,
                args.output_dir,
                camera_id=args.camera_id,
            )
            print(f"Read {count} operator_verified projections")
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(2, f"Cannot prepare geometry review: {error}\n")


if __name__ == "__main__":
    main()

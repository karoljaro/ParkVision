"""Verify whether an approved ROI projection still applies in native pixels."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from ..configuration import ParkingConfiguration
from ..detection.frame_registration import Registration, register_background

IntegrityStatus = Literal["verified", "mismatch", "unverifiable"]
# This is a limit on using the same ROI coordinates, not a registration
# acceptance threshold. Phase 4C's wider limits allow alignment for diagnosis.
MAX_NATIVE_ROI_DISPLACEMENT_PX = 1.5
ROI_EXCLUSION_PX = 24
MASK_EDGE_MARGIN_PX = 8


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    status: IntegrityStatus
    reason: str
    reference_sha256: str
    manifest_sha256: str
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CameraIntegrityReference:
    """Operator-approved reference for one camera and exact geometry revision."""

    camera_id: str
    configuration_revision: int
    configuration_sha256: str
    image_size: tuple[int, int]
    source: str
    approved_by: str
    approved_at: str
    reference_sha256: str
    manifest_sha256: str
    image: NDArray[np.uint8] = field(repr=False, compare=False)
    fit_mask: NDArray[np.uint8] = field(repr=False, compare=False)
    check_mask: NDArray[np.uint8] = field(repr=False, compare=False)
    roi_vertices: NDArray[np.float64] = field(repr=False, compare=False)

    @classmethod
    def load(
        cls,
        path: Path,
        configuration: ParkingConfiguration,
        configuration_sha256: str,
        camera_id: str,
    ) -> "CameraIntegrityReference":
        raw = path.read_bytes()
        data = json.loads(raw)
        required = {
            "schema_version",
            "camera_id",
            "configuration_sha256",
            "configuration_revision",
            "image_size",
            "source",
            "approved_by",
            "approved_at",
            "reference",
            "fit_mask",
            "check_mask",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Invalid camera integrity reference manifest")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported camera reference schema")
        if (
            not configuration.is_approved
            or camera_id not in configuration.camera_image_sizes
            or data["camera_id"] != camera_id
            or type(data["configuration_revision"]) is not int
            or data["configuration_revision"] != configuration.revision
            or data["configuration_sha256"] != configuration_sha256
            or not _is_sha256(configuration_sha256)
        ):
            raise ValueError(
                "Camera reference does not bind the approved configuration"
            )
        width, height = configuration.camera_image_sizes[camera_id]
        size = data["image_size"]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(value) is not int for value in size)
            or size != [width, height]
        ):
            raise ValueError("Camera reference size does not match configuration")
        for key in ("source", "approved_by", "approved_at"):
            value = data[key]
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"Camera reference requires an approved {key}")
        if datetime.fromisoformat(data["approved_at"]).utcoffset() is None:
            raise ValueError("Camera reference approval must include a UTC offset")

        image = _decode_image(
            path.parent,
            data["reference"],
            (height, width, 3),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        masks = tuple(
            _decode_image(path.parent, data[key], (height, width), cv2.IMREAD_UNCHANGED)
            for key in ("fit_mask", "check_mask")
        )
        fit, check = masks
        if any(not np.isin(mask, (0, 255)).all() for mask in masks):
            raise ValueError("Static masks must contain only 0/255 grayscale pixels")
        if np.any((fit != 0) & (check != 0)):
            raise ValueError("Static fit and check masks must be disjoint")

        projections = configuration.projections_for_camera(
            camera_id, image_width=width, image_height=height
        )
        if not projections:
            raise ValueError("Camera integrity needs ROI projections for the camera")
        polygons = [
            np.float64(
                [
                    (point.x * (width - 1), point.y * (height - 1))
                    for point in projection.polygon.vertices
                ]
            )
            for projection in projections
        ]
        roi_vertices = np.concatenate(polygons)
        exclusion = np.zeros((height, width), np.uint8)
        for polygon in polygons:
            cv2.fillPoly(exclusion, [np.int32(np.round(polygon))], 255)
        exclusion = cv2.dilate(
            exclusion,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ROI_EXCLUSION_PX + 1,) * 2
            ),
        )
        for mask in masks:
            mask[exclusion != 0] = 0
            if not np.any(_interior(mask)):
                raise ValueError("Static masks need usable landmarks outside ROI")
            mask.flags.writeable = False
        image.flags.writeable = False
        roi_vertices.flags.writeable = False
        return cls(
            camera_id,
            configuration.revision,
            configuration_sha256,
            (width, height),
            data["source"],
            data["approved_by"],
            data["approved_at"],
            data["reference"]["sha256"],
            sha256(raw).hexdigest(),
            image,
            fit,
            check,
            roi_vertices,
        )

    def check(self, image: NDArray[np.uint8]) -> IntegrityResult:
        """Return evidence only. Fitted transforms do not change ROI or pixels."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            return self._result("unverifiable", "invalid_image")
        if image.shape != self.image.shape:
            return self._result("mismatch", "resolution_mismatch")

        result = register_background(
            self.image,
            image,
            _interior(self.fit_mask),
            validation_mask=_interior(self.check_mask),
        )
        diagnostics = _registration_diagnostics(result)
        failures = set(result.failure_reasons)
        if result.matrix is None or (
            not result.accepted and failures != {"camera_view_changed"}
        ):
            return self._result("unverifiable", result.reason, diagnostics)

        fit_points = []
        check_points = []
        for point in result.correspondences:
            if point["role"] == "fit":
                if not point["inlier"]:
                    continue
                mask = self.fit_mask
                fit_points.append(point["reference"])
            elif point["group"] == "additional_mask":
                mask = self.check_mask
                check_points.append(point["reference"])
            else:
                mask = self.fit_mask
            if not all(
                _inside_mask(mask, point[key]) for key in ("source", "reference")
            ):
                return self._result(
                    "unverifiable", "matching_left_static_masks", diagnostics
                )
        fit_covers = _covers_roi(fit_points, self.roi_vertices)
        check_covers = _covers_roi(check_points, self.roi_vertices)
        diagnostics.update(
            fit_covers_roi=fit_covers,
            independent_check_covers_roi=check_covers,
        )
        if not fit_covers or not check_covers:
            return self._result(
                "unverifiable", "insufficient_roi_spatial_support", diagnostics
            )

        matrix = np.float64(result.matrix)
        inverse = cv2.invertAffineTransform(matrix)
        shifted = self.roi_vertices @ inverse[:, :2].T + inverse[:, 2]
        displacement = float(
            np.max(np.linalg.norm(shifted - self.roi_vertices, axis=1))
        )
        diagnostics["max_roi_displacement_px"] = displacement
        if not result.accepted or displacement > MAX_NATIVE_ROI_DISPLACEMENT_PX:
            return self._result("mismatch", "camera_view_changed", diagnostics)
        return self._result("verified", "native_roi_alignment_verified", diagnostics)

    def _result(self, status: IntegrityStatus, reason: str, diagnostics=None):
        return IntegrityResult(
            status,
            reason,
            self.reference_sha256,
            self.manifest_sha256,
            diagnostics if diagnostics is not None else {},
        )


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _decode_image(directory, record, shape, flags):
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or not isinstance(record["path"], str)
        or not record["path"].strip()
        or not _is_sha256(record["sha256"])
    ):
        raise ValueError("Expected a pinned camera reference image or mask")
    path = directory / record["path"]
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Camera reference or mask SHA-256 mismatch")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    if image is None or image.shape != shape or image.dtype != np.uint8:
        raise ValueError("Camera reference or mask has invalid pixels or dimensions")
    return image


def _interior(mask):
    # Leave margin for Phase 4C's native corner refinement.
    return cv2.erode(mask, np.ones((2 * MASK_EDGE_MARGIN_PX + 1,) * 2, np.uint8))


def _inside_mask(mask, point):
    x, y = (round(value) for value in point)
    height, width = mask.shape
    return 0 <= x < width and 0 <= y < height and mask[y, x] != 0


def _covers_roi(points, roi_vertices):
    if len(points) < 6:
        return False
    hull = cv2.convexHull(np.float32(points))
    return all(
        cv2.pointPolygonTest(hull, tuple(vertex), False) >= 0 for vertex in roi_vertices
    )


def _registration_diagnostics(result: Registration) -> dict:
    diagnostics = {
        "method": result.method,
        "registration_reason": result.reason,
        "registration_failures": result.failure_reasons,
        "fit_inliers": result.inliers,
        "fit_coverage": result.coverage,
        "fit_cells": result.occupied_cells,
        "fit_p90_px": result.residual_px,
        "validation_count": result.validation_count,
        "validation_coverage": result.validation_coverage,
        "validation_cells": result.validation_cells,
        "validation_p90_px": result.validation_p90_px,
        "validation_max_px": result.validation_max_px,
        "parameter_stability_px": result.parameter_stability_px,
        "transform_source_to_reference": result.matrix,
        "max_native_roi_displacement_px": MAX_NATIVE_ROI_DISPLACEMENT_PX,
    }
    if result.matrix is not None:
        (a, _, dx), (b, _, dy) = result.matrix
        diagnostics.update(
            scale=hypot(a, b),
            rotation_degrees=degrees(atan2(b, a)),
            translation_px=(dx, dy),
        )
    return diagnostics

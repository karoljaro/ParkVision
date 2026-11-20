"""Inspectable registration correspondences and candidate edge alignment."""

import cv2
import numpy as np


def render_registration_preview(reference, image, registration: dict) -> np.ndarray:
    width = 720
    ratio = min(width / reference.shape[1], 480 / reference.shape[0])
    size = (round(reference.shape[1] * ratio), round(reference.shape[0] * ratio))
    height = size[1]

    def point(value):
        return tuple(round(float(v) * ratio) for v in value)

    ref = cv2.resize(reference, size)
    src = cv2.resize(image, size)
    pair = np.full((height, 2 * width, 3), 28, np.uint8)
    pair[:, : size[0]] = ref
    pair[:, width : width + size[0]] = src
    correspondences = registration.get("correspondences", ())
    # Full records remain in JSON. Limit drawing density without hiding failures.
    stride = max(1, len(correspondences) // 300)
    for index, item in enumerate(correspondences):
        if index % stride and item["inlier"]:
            continue
        target = point(item["reference"])
        source = point(item["source"])
        source = (source[0] + width, source[1])
        color = (
            (255, 210, 0)
            if item["role"] == "check"
            else (40, 220, 40) if item["inlier"] else (30, 30, 240)
        )
        cv2.line(pair, target, source, color, 1, cv2.LINE_AA)
        cv2.circle(pair, target, 3, color, 1)
        cv2.circle(pair, source, 3, color, 1)
    matrix = registration.get("matrix")
    aligned = (
        image
        if matrix is None
        else cv2.warpAffine(
            image, np.array(matrix), (reference.shape[1], reference.shape[0])
        )
    )

    def edges(source):
        first = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 80, 160)
        second = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), 80, 160)
        composite = np.stack((second, second, first), axis=2)
        return cv2.resize(composite, size)

    comparison = np.full_like(pair, 28)
    comparison[:, : size[0]] = edges(image)
    comparison[:, width : width + size[0]] = edges(aligned)
    # A separate reference plot shows the actual spatial distribution and errors.
    spatial = cv2.addWeighted(ref, 0.45, np.zeros_like(ref), 0.55, 0)
    for col in range(1, 4):
        x = round(size[0] * col / 4)
        cv2.line(spatial, (x, 0), (x, height), (100, 100, 100), 1)
    for row in range(1, 3):
        y = round(height * row / 3)
        cv2.line(spatial, (0, y), (size[0], y), (100, 100, 100), 1)
    for item in correspondences:
        color = (
            (255, 210, 0)
            if item["role"] == "check"
            else (40, 220, 40) if item["inlier"] else (30, 30, 240)
        )
        cv2.circle(spatial, point(item["reference"]), 2, color, -1)
    summary = np.full_like(pair, 28)
    summary[:, : size[0]] = spatial
    lines = [
        f"accepted={registration['accepted']}  {registration['reason']}",
        f"method: {registration.get('method', 'legacy_ORB')}",
        f"fit: {registration['inliers']}/{registration['matches']} inliers",
        f"fit p50/p90/max: {registration.get('residual_p50_px')} /",
        f"  {registration.get('residual_px')} / "
        f"{registration.get('residual_max_px')} px",
        f"fit hull/cells: {registration.get('coverage', 0):.3f} / "
        f"{registration.get('occupied_cells')}",
        f"unused checks: {registration.get('validation_count', 0)}",
        f"check p90/max: {registration.get('validation_p90_px')} /",
        f"  {registration.get('validation_max_px')} px",
        f"check hull/cells: {registration.get('validation_coverage', 0):.3f} / "
        f"{registration.get('validation_cells')}",
        "green=fit inlier; red=fit outlier; cyan=unused check",
        "Edges red=reference, cyan=source; moving cars differ.",
        "Candidate transform may be rejected. Inspect JSON reasons.",
    ]
    for row, line in enumerate(lines):
        cv2.putText(
            summary,
            line[:100],
            (width + 10, 22 + row * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def caption(title):
        strip = np.full((38, 2 * width, 3), 28, np.uint8)
        cv2.putText(
            strip,
            title,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return strip

    return np.vstack(
        (
            caption("Reference / source correspondences"),
            pair,
            caption("Raw edge disagreement / candidate-aligned disagreement"),
            comparison,
            caption("Spatial coverage and full-resolution alignment errors"),
            summary,
        )
    )

"""Bounded registration on explicitly selected, stationary background only."""

from dataclasses import dataclass
from math import atan2, degrees, hypot

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Registration:
    accepted: bool
    reason: str
    matrix: tuple[tuple[float, ...], ...] | None = None
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    residual_px: float | None = None
    coverage: float = 0.0
    occupied_cells: int = 0


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def register_background(
    reference: NDArray[np.uint8],
    image: NDArray[np.uint8],
    background_mask: NDArray[np.uint8] | None,
) -> Registration:
    """Reject uncertain, zoomed or moved views; never estimate a homography.

    The mask must select stationary scenery in the reference AND source views,
    excluding parking surfaces, vehicles and vegetation. Broadly distributed
    inliers are required even when a mask is supplied. This is an explicit
    operator assertion, not automatic semantic recognition of the background.
    """
    if reference.shape != image.shape:
        return Registration(False, "resolution_mismatch")
    if background_mask is None:
        return Registration(False, "stationary_background_mask_required")
    height, width = reference.shape[:2]
    if background_mask.shape != (height, width) or background_mask.dtype != np.uint8:
        raise ValueError("Background mask must be uint8 with reference dimensions")
    # A bounded downscale limits cost; all reported errors remain full-size pixels.
    ratio = min(1.0, 1200 / max(height, width))
    size = (round(width * ratio), round(height * ratio))
    ref = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), size)
    src = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size)
    mask = cv2.resize(background_mask, size, interpolation=cv2.INTER_NEAREST)
    orb = cv2.ORB_create(nfeatures=4500, fastThreshold=12)
    rk, rd = orb.detectAndCompute(ref, mask)
    sk, sd = orb.detectAndCompute(src, mask)
    if rd is None or sd is None or len(rd) < 25 or len(sd) < 25:
        return Registration(False, "insufficient_background_features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(sd, rd, k=2)
    backward = matcher.knnMatch(rd, sd, k=2)
    reverse = {
        a.queryIdx: a.trainIdx
        for entry in backward
        if len(entry) == 2
        for a, b in [entry]
        if a.distance < 0.75 * b.distance
    }
    matches = [
        a
        for entry in forward
        if len(entry) == 2
        for a, b in [entry]
        if a.distance < 0.75 * b.distance and reverse.get(a.trainIdx) == a.queryIdx
    ]
    if len(matches) < 25:
        return Registration(False, "insufficient_mutual_matches", matches=len(matches))
    source = np.float32([sk[m.queryIdx].pt for m in matches]) / ratio
    target = np.float32([rk[m.trainIdx].pt for m in matches]) / ratio
    matrix, selected = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=3000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None or selected is None:
        return Registration(False, "background_fit_failed", matches=len(matches))
    good = selected.ravel().astype(bool)
    count = int(good.sum())
    fraction = count / len(matches)
    points = target[good]
    coverage = (
        cv2.contourArea(cv2.convexHull(points)) / (width * height) if count >= 3 else 0
    )
    cells = {
        (min(3, int(x * 4 / width)), min(2, int(y * 3 / height))) for x, y in points
    }
    errors = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1)
    residual = float(np.quantile(errors[good], 0.9)) if count else None
    scale = hypot(matrix[0, 0], matrix[1, 0])
    angle = abs(degrees(atan2(matrix[1, 0], matrix[0, 0])))
    shift = hypot(matrix[0, 2], matrix[1, 2])
    reason = "background_aligned"
    if count < 25 or fraction < 0.60 or residual > 1.5:
        reason = "unreliable_background_fit"
    elif coverage < 0.15 or len(cells) < 6:
        reason = "insufficient_spatial_coverage"
    elif (
        abs(scale - 1) > 0.005
        or angle > 0.75
        or shift > max(8, 0.012 * hypot(width, height))
    ):
        reason = "camera_view_changed"
    return Registration(
        reason == "background_aligned",
        reason,
        tuple(tuple(float(v) for v in row) for row in matrix),
        len(matches),
        count,
        fraction,
        residual,
        float(coverage),
        len(cells),
    )

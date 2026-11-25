"""Bounded similarity registration with native refinement and withheld checks."""

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, ceil, cos, degrees, hypot, radians, sqrt

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
    method: str = "orb_native_holdout"
    residual_p50_px: float | None = None
    residual_max_px: float | None = None
    validation_p50_px: float | None = None
    validation_p90_px: float | None = None
    validation_max_px: float | None = None
    validation_count: int = 0
    validation_coverage: float = 0.0
    validation_cells: int = 0
    parameter_stability_px: float | None = None
    correspondences: tuple[dict, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    raw_matches: int = 0
    refined_matches: int = 0
    provenance: str = (
        "automatic_spatial_holdout_same_features_not_independent_ground_truth"
    )
    appearance_min_ncc: float | None = None
    validation_groups: tuple[dict, ...] = ()


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
MAX_SCALE_CHANGE = 0.005
MAX_ROTATION_DEGREES = 0.75
CORNER_REFINEMENT_RADIUS_PX = 3


def _translation_limit(width, height):
    return max(8, 0.012 * hypot(width, height))


def _refinement_shift_limit(ratio):
    return max(4, 4 / ratio)


def _coverage(points, width, height):
    hull = cv2.convexHull(np.float32(points)) if len(points) >= 3 else None
    area = cv2.contourArea(hull) / (width * height) if hull is not None else 0.0
    cells = {
        (min(3, int(x * 4 / width)), min(2, int(y * 3 / height))) for x, y in points
    }
    return float(area), len(cells)


def _similarity(source, target):
    """Least squares on ALL supplied pairs, without rejecting manual outliers."""
    center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    a, b = source - center, target - target_center
    denominator = float(np.sum(a * a))
    if denominator < 1e-9:
        raise ValueError("Degenerate control points")
    real = float(np.sum(a * b)) / denominator
    imag = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])) / denominator
    rotation = np.array([[real, -imag], [imag, real]])
    return np.column_stack((rotation, target_center - rotation @ center))


def _errors(matrix, source, target):
    return np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1)


def _patch_score(reference, image, source, target, size=15):
    height, width = reference.shape
    margin = size // 2 + 1
    if not all(
        margin <= p[0] < width - margin and margin <= p[1] < height - margin
        for p in (source, target)
    ):
        return -1.0
    first = cv2.getRectSubPix(
        reference, (size, size), tuple(float(v) for v in target)
    ).astype(float)
    second = cv2.getRectSubPix(
        image, (size, size), tuple(float(v) for v in source)
    ).astype(float)
    if min(first.std(), second.std()) < 3:
        return -1.0
    first -= first.mean()
    second -= second.mean()
    return float(
        np.sum(first * second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )


def _evaluate(
    source,
    target,
    is_check,
    matrix,
    good,
    shape,
    *,
    method,
    raw_matches,
    provenance,
    ids=(),
    patch_scores=None,
    check_groups=None,
):
    height, width = shape[:2]
    errors = _errors(matrix, source, target)
    fit = ~is_check
    supported = fit & good
    fit_errors = errors[supported]
    check_errors = errors[is_check]
    count = int(supported.sum())
    matches = int(fit.sum())
    coverage, cells = _coverage(target[supported], width, height)
    check_coverage, check_cells = _coverage(target[is_check], width, height)
    p50, p90, maximum = (
        tuple(float(x) for x in np.quantile(fit_errors, [0.5, 0.9, 1]))
        if count
        else (None, None, None)
    )
    c50, c90, cmax = (
        tuple(float(x) for x in np.quantile(check_errors, [0.5, 0.9, 1]))
        if len(check_errors)
        else (None, None, None)
    )
    corners = np.float64(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
    )
    stability = 0.0
    # Leave one spatial cell out of the FIT set only. Validation never refits it.
    fit_cells = np.minimum(np.int32(target[:, 0] * 4 / width), 3) + 4 * np.minimum(
        np.int32(target[:, 1] * 3 / height), 2
    )
    for cell in set(fit_cells[supported]):
        subset = supported & (fit_cells != cell)
        if subset.sum() < 4:
            stability = float("inf")
            break
        alternative = _similarity(source[subset], target[subset])
        stability = max(
            stability,
            float(
                np.max(
                    np.linalg.norm(
                        corners @ (alternative[:, :2] - matrix[:, :2]).T
                        + alternative[:, 2]
                        - matrix[:, 2],
                        axis=1,
                    )
                )
            ),
        )
    failures = []
    if patch_scores is not None and min(patch_scores, default=-1) < 0.85:
        failures.append("control_point_appearance_mismatch")
    minimum = 6 if method == "control_points" else 25
    if count < minimum or count / max(1, matches) < 0.60 or p90 is None or p90 > 1.5:
        failures.append("unreliable_background_fit")
    if method == "control_points" and (maximum is None or maximum > 3):
        failures.append("inconsistent_control_points")
    if coverage < 0.15 or cells < 6:
        failures.append("insufficient_spatial_coverage")
    check_minimum = 6 if method == "control_points" else 12
    if len(check_errors) < check_minimum or check_coverage < 0.15 or check_cells < 6:
        failures.append("insufficient_validation_coverage")
    if c90 is None or c90 > 1.5 or cmax is None or cmax > 3:
        failures.append("held_out_alignment_failed")
    groups = []
    if check_groups is not None:
        for name in sorted(set(check_groups[is_check])):
            selected = is_check & (check_groups == name)
            values = errors[selected]
            area, occupied = _coverage(target[selected], width, height)
            groups.append(
                {
                    "group": name,
                    "count": int(selected.sum()),
                    "p50_px": float(np.median(values)),
                    "p90_px": float(np.quantile(values, 0.9)),
                    "max_px": float(values.max()),
                    "coverage": area,
                    "cells": occupied,
                }
            )
            if name == "additional_mask" and selected.sum() < 6:
                failures.append("insufficient_additional_validation")
            if name == "spatial_holdout" and (
                selected.sum() < 12 or area < 0.15 or occupied < 6
            ):
                failures.append("insufficient_validation_coverage")
            if float(np.quantile(values, 0.9)) > 1.5 or float(values.max()) > 3:
                failures.append("held_out_alignment_failed")
    if not np.isfinite(stability) or stability > 1.5:
        failures.append("unstable_transform")
    scale = hypot(matrix[0, 0], matrix[1, 0])
    angle = abs(degrees(atan2(matrix[1, 0], matrix[0, 0])))
    shift = hypot(matrix[0, 2], matrix[1, 2])
    if (
        abs(scale - 1) > MAX_SCALE_CHANGE
        or angle > MAX_ROTATION_DEGREES
        or shift > _translation_limit(width, height)
    ):
        failures.append("camera_view_changed")
    records = tuple(
        {
            "id": str(ids[i]) if ids else str(i),
            "source": tuple(float(v) for v in a),
            "reference": tuple(float(v) for v in b),
            "role": "check" if is_check[i] else "fit",
            "error_px": float(errors[i]),
            "inlier": bool(good[i]),
            "group": (
                str(check_groups[i]) if check_groups is not None else "control_points"
            ),
        }
        for i, (a, b) in enumerate(zip(source, target, strict=True))
    )
    return Registration(
        accepted=not failures,
        reason=failures[0] if failures else "background_aligned",
        matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        matches=matches,
        inliers=count,
        inlier_ratio=count / max(1, matches),
        residual_px=p90,
        coverage=coverage,
        occupied_cells=cells,
        method=method,
        residual_p50_px=p50,
        residual_max_px=maximum,
        validation_p50_px=c50,
        validation_p90_px=c90,
        validation_max_px=cmax,
        validation_count=len(check_errors),
        validation_coverage=check_coverage,
        validation_cells=check_cells,
        parameter_stability_px=stability if np.isfinite(stability) else None,
        correspondences=records,
        failure_reasons=tuple(failures),
        raw_matches=raw_matches,
        refined_matches=len(source),
        provenance=provenance,
        appearance_min_ncc=min(patch_scores) if patch_scores is not None else None,
        validation_groups=tuple(groups),
    )


def _control_registration(reference, image, controls):
    if (
        not isinstance(controls, dict)
        or not isinstance(controls.get("provenance"), str)
        or not controls["provenance"].strip()
    ):
        raise ValueError("Control points require nonempty provenance")
    for key, pixels in [
        ("reference_pixel_sha256", reference),
        ("source_pixel_sha256", image),
    ]:
        if controls.get(key) != sha256(pixels.tobytes()).hexdigest():
            raise ValueError("Control point pixel hash mismatch")
    height, width = reference.shape[:2]
    source, target, roles, ids = [], [], [], []
    for role in ("fit", "check"):
        entries = controls.get(role)
        if not isinstance(entries, list) or len(entries) < 6:
            raise ValueError("At least six fit and six check points are required")
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("id"), str)
                or not entry["id"].strip()
            ):
                raise ValueError("Control point IDs must be nonempty strings")
            ids.append(entry["id"])
            for key, collection in [("source", source), ("reference", target)]:
                point = np.asarray(entry.get(key), dtype=np.float64)
                if (
                    point.shape != (2,)
                    or not np.isfinite(point).all()
                    or not (0 <= point[0] < width and 0 <= point[1] < height)
                ):
                    raise ValueError(
                        "Control coordinates must be finite, in-frame pixel centers"
                    )
                collection.append(point)
            roles.append(role == "check")
    if len(ids) != len(set(ids)):
        raise ValueError("Control point IDs must be unique")
    source, target, is_check = np.array(source), np.array(target), np.array(roles)
    for points in (source, target):
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, float("inf"))
        if np.min(distances) < 12:
            raise ValueError("Use distinct controls separated by at least 12 pixels")
        for selected in (is_check, ~is_check):
            area, cells = _coverage(points[selected], width, height)
            if area < 0.15 or cells < 6:
                raise ValueError("Control points have insufficient spatial coverage")
    matrix = _similarity(source[~is_check], target[~is_check])
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patch_scores = [
        _patch_score(gray, src_gray, a, b) for a, b in zip(source, target, strict=True)
    ]
    return _evaluate(
        source,
        target,
        is_check,
        matrix,
        np.ones(len(source), bool),
        reference.shape,
        method="control_points",
        raw_matches=len(source),
        provenance=controls["provenance"],
        ids=ids,
        patch_scores=patch_scores,
    )


def _orb_candidate_mask(source, target, width, height, ratio):
    # A bounded similarity can move each point by at most the translation limit
    # plus the effect of scale and rotation at its distance from the origin.
    angle = radians(MAX_ROTATION_DEGREES)
    relative_motion = sqrt(
        MAX_SCALE_CHANGE**2 + 2 * (1 + MAX_SCALE_CHANGE) * (1 - cos(angle))
    )
    # ORB observes initial locations, before native corner and flow refinement.
    refinement_margin = _refinement_shift_limit(ratio) + ceil(
        hypot(CORNER_REFINEMENT_RADIUS_PX, CORNER_REFINEMENT_RADIUS_PX)
    )
    radius = (
        _translation_limit(width, height)
        + relative_motion * np.linalg.norm(source, axis=1)
        + refinement_margin
    )
    allowed = np.empty((len(source), len(target)), dtype=np.uint8)
    # Keep the mask dense for OpenCV but bound temporary coordinate arrays.
    for start in range(0, len(source), 128):
        block = slice(start, start + 128)
        delta = source[block, None, :] - target[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
        np.less_equal(distance_squared, radius[block, None] ** 2, out=allowed[block])
    return allowed


def _orb_matches(reference, image, mask, *, limit_displacement=True):
    height, width = reference.shape
    ratio = min(1.0, 1200 / max(height, width))
    size = (round(width * ratio), round(height * ratio))
    ref, src = cv2.resize(reference, size), cv2.resize(image, size)
    selected = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    # A 31-pixel exclusion at reduced resolution discarded real static fence
    # corners. Use the descriptor's half-width; native patch/FB checks still
    # reject unsupported border matches before fitting.
    orb = cv2.ORB_create(nfeatures=4500, fastThreshold=12, edgeThreshold=15)
    rk, rd = orb.detectAndCompute(ref, selected)
    sk, sd = orb.detectAndCompute(src, selected)
    if rd is None or sd is None or len(rd) < 2 or len(sd) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), 0.0
    # OpenCV resize maps pixel centers. Do not magnify its half-pixel offset.
    factor = np.array([width / size[0], height / size[1]])
    target = (np.float64([key.pt for key in rk]) + 0.5) * factor - 0.5
    source = (np.float64([key.pt for key in sk]) + 0.5) * factor - 0.5
    allowed = (
        _orb_candidate_mask(source, target, width, height, ratio)
        if limit_displacement
        else None
    )
    reverse_allowed = None if allowed is None else np.ascontiguousarray(allowed.T)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    backward = matcher.knnMatch(rd, sd, k=2, mask=reverse_allowed)
    reverse = {
        a.queryIdx: a.trainIdx
        for entry in backward
        if len(entry) == 2
        for a, b in [entry]
        if a.distance < 0.75 * b.distance
    }
    forward = matcher.knnMatch(sd, rd, k=2, mask=allowed)
    matches = [
        a
        for entry in forward
        if len(entry) == 2
        for a, b in [entry]
        if a.distance < 0.75 * b.distance and reverse.get(a.trainIdx) == a.queryIdx
    ]
    return (
        source[[match.queryIdx for match in matches]],
        target[[match.trainIdx for match in matches]],
        ratio,
    )


def _quadratic_peak_offset(scores, x, y):
    """Locate the subpixel peak of the local two-dimensional quadratic."""
    center = scores[y, x]
    dx = (scores[y, x + 1] - scores[y, x - 1]) / 2
    dy = (scores[y + 1, x] - scores[y - 1, x]) / 2
    xx = scores[y, x + 1] + scores[y, x - 1] - 2 * center
    yy = scores[y + 1, x] + scores[y - 1, x] - 2 * center
    xy = (
        scores[y + 1, x + 1]
        - scores[y + 1, x - 1]
        - scores[y - 1, x + 1]
        + scores[y - 1, x - 1]
    ) / 4
    determinant = xx * yy - xy * xy
    if xx >= 0 or determinant <= 0:
        return None
    offset = np.array([(xy * dy - yy * dx), (xy * dx - xx * dy)]) / determinant
    return offset if np.max(np.abs(offset)) <= 1 else None


def _normalized_patch(image, point, size):
    patch = cv2.getRectSubPix(image, (size, size), tuple(point)).ravel()
    patch -= patch.mean()
    norm = np.linalg.norm(patch)
    return patch / max(float(norm), 1e-6)


def _fit_subpixel_patch(reference, image, point, position, size):
    # Compare actual pixels in the sample with an interpolated reference.
    # Resampling both sides or resampling an already softened source is biased
    # in the immediate neighborhood of an integer translation.
    sample_center = np.rint(position)
    center = sample_center - position + point
    sample = _normalized_patch(image, sample_center, size)
    offset = np.zeros(2)
    epsilon = 1 / 32
    for _ in range(4):
        estimate = _normalized_patch(reference, center - offset, size)
        residual = sample - estimate
        derivatives = np.column_stack(
            [
                (
                    _normalized_patch(reference, center - offset - delta, size)
                    - _normalized_patch(reference, center - offset + delta, size)
                )
                / (2 * epsilon)
                for delta in np.eye(2) * epsilon
            ]
        )
        step, _, rank, _ = np.linalg.lstsq(derivatives, residual, rcond=None)
        if rank < 2 or np.linalg.norm(offset + step) > 0.75:
            return None
        offset += step
        if np.linalg.norm(step) < 0.002:
            break
    return position + offset


def _match_normalized_patch(reference, image, point, seed, limit, radius, grid):
    size = 21
    height, width = image.shape
    # Sample both images at the same subpixel phase for integer translations.
    center = point + np.rint(seed - point)
    for (x, y), margin in ((point, size // 2 + 1), (center, size // 2 + radius + 1)):
        if not (margin <= x < width - margin and margin <= y < height - margin):
            return None

    template = cv2.getRectSubPix(reference, (size, size), tuple(point))
    contrast = float(template.std())
    if contrast < 3:
        return None
    search_size = size + 2 * radius
    search = cv2.getRectSubPix(image, (search_size, search_size), tuple(center))
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (x, y) = cv2.minMaxLoc(scores)
    if peak < 0.85 or not (0 < x < 2 * radius and 0 < y < 2 * radius):
        return None

    # Compare the best peak with other plausible locations in the search.
    # After fitting gain and offset, the RMS is std(template) * sqrt(1 - NCC²).
    # Two locations within one 8-bit intensity level of each other are ambiguous.
    far = (grid - x) ** 2 + (grid[:, None] - y) ** 2 >= 4
    competing = max(0.0, float(scores[far].max()))
    best_rms = sqrt(max(0, 1 - peak**2))
    other_rms = sqrt(max(0, 1 - competing**2))
    offset = _quadratic_peak_offset(scores, x, y)
    if offset is None or contrast * (other_rms - best_rms) < 1.0:
        return None

    integer_position = center + np.array([x - radius, y - radius])
    position = integer_position + offset
    score = _patch_score(reference, image, position, point, size)
    integer_score = _patch_score(reference, image, integer_position, point, size)
    if integer_score > score:
        position, score = integer_position, integer_score
    if np.linalg.norm(position - seed) > limit or score < 0.85:
        return None

    refined = _fit_subpixel_patch(reference, image, point, position, size)
    if refined is None or np.linalg.norm(refined - seed) > limit:
        return None
    if _patch_score(reference, image, refined, point, size) < 0.85:
        return None
    return refined


def _refine(reference, image, source, target, ratio):
    if not len(source):
        return source, target
    target = np.float32(target).reshape(-1, 1, 2)
    cv2.cornerSubPix(
        reference,
        target,
        (CORNER_REFINEMENT_RADIUS_PX, CORNER_REFINEMENT_RADIUS_PX),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    reference, image = reference.astype(np.float32), image.astype(np.float32)
    limit = _refinement_shift_limit(ratio)
    # Include a border for peak interpolation and the half-pixel grid offset.
    radius = ceil(limit + 0.5) + 1
    grid = np.arange(2 * radius + 1)
    matched_source, matched_target = [], []
    for seed, point in zip(source, target.reshape(-1, 2), strict=True):
        match = _match_normalized_patch(
            reference, image, point, seed, limit, radius, grid
        )
        if match is None:
            continue
        back = _match_normalized_patch(
            image, reference, match, point, limit, radius, grid
        )
        if back is None or np.linalg.norm(back - point) > 0.75:
            continue
        if _patch_score(reference, image, match, point, 11) < 0.85:
            continue
        matched_source.append(match)
        matched_target.append(point)
    return (
        np.float64(matched_source).reshape(-1, 2),
        np.float64(matched_target).reshape(-1, 2),
    )


def _distinct_points(source, target, minimum=12.0):
    """ORB scale levels must not put the same physical corner in fit and check."""
    selected = []
    grids = ({}, {})
    for index in range(len(source)):
        keys = [
            tuple(int(v // minimum) for v in point)
            for point in (source[index], target[index])
        ]
        nearby = False
        for points, grid, key in zip((source, target), grids, keys, strict=True):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if any(
                        np.linalg.norm(points[index] - points[other]) < minimum
                        for other in grid.get((key[0] + dx, key[1] + dy), ())
                    ):
                        nearby = True
        if nearby:
            continue
        selected.append(index)
        for grid, key in zip(grids, keys, strict=True):
            grid.setdefault(key, []).append(index)
    return source[selected], target[selected]


def register_background(
    reference: NDArray[np.uint8],
    image: NDArray[np.uint8],
    background_mask: NDArray[np.uint8] | None,
    *,
    validation_mask: NDArray[np.uint8] | None = None,
    control_points: dict | None = None,
) -> Registration:
    """Fit a bounded similarity and check unused, distributed static points.

    Masks may include deliberately reviewed permanent pavement markings/curbs,
    never unconditionally the parking surface. Automatic holdout uses the same
    feature method and is explicitly NOT independent ground truth. Supplied
    controls are optional and never bypass validation or camera-motion limits.
    """
    if reference.shape != image.shape:
        return Registration(False, "resolution_mismatch")
    if control_points is not None:
        try:
            return _control_registration(reference, image, control_points)
        except (ValueError, TypeError, KeyError) as error:
            return Registration(
                False,
                "invalid_control_points",
                method="control_points",
                failure_reasons=(str(error),),
            )
    if background_mask is None:
        return Registration(False, "stationary_background_mask_required")
    height, width = reference.shape[:2]
    for mask in (background_mask, validation_mask):
        if mask is not None and (
            mask.shape != (height, width) or mask.dtype != np.uint8
        ):
            raise ValueError("Background mask must be uint8 with reference dimensions")
    if validation_mask is not None and np.any(
        (background_mask > 0) & (validation_mask > 0)
    ):
        raise ValueError("Fit and validation masks must be disjoint")
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = _register_automatic(
        gray, src_gray, background_mask, validation_mask, limit_displacement=True
    )
    if result.accepted or result.failure_reasons == ("camera_view_changed",):
        return result
    # Points can move beyond the local search after a real camera change.
    # Global matches may diagnose that change only when every other check holds.
    diagnostic = _register_automatic(
        gray, src_gray, background_mask, validation_mask, limit_displacement=False
    )
    return (
        diagnostic if diagnostic.failure_reasons == ("camera_view_changed",) else result
    )


def _register_automatic(
    gray, src_gray, background_mask, validation_mask, *, limit_displacement
):
    height, width = gray.shape
    source, target, ratio = _orb_matches(
        gray, src_gray, background_mask, limit_displacement=limit_displacement
    )
    raw = len(source)
    if raw < 25:
        return Registration(
            False,
            (
                "insufficient_background_features"
                if ratio == 0
                else "insufficient_mutual_matches"
            ),
            matches=raw,
            raw_matches=raw,
        )
    source, target = _refine(gray, src_gray, source, target, ratio)
    source, target = _distinct_points(source, target)
    # Always withhold spatially interleaved points, even with extra check regions.
    # Extra regions ADD checks; they never replace or weaken the global holdout.
    ordered = sorted(
        range(len(target)),
        key=lambda i: (
            int(target[i, 1] * 3 / height),
            int(target[i, 0] * 4 / width),
            target[i, 1],
            target[i, 0],
        ),
    )
    check = np.zeros(len(source), bool)
    cell_counts = {}
    for index in ordered:
        cell = (int(target[index, 0] * 4 / width), int(target[index, 1] * 3 / height))
        count = cell_counts.get(cell, 0)
        check[index] = count % 3 == 0
        cell_counts[cell] = count + 1
    groups = np.array(
        ["spatial_holdout" if value else "fit" for value in check], dtype=object
    )
    provenance = "automatic_spatial_holdout_same_features_not_independent_ground_truth"
    if validation_mask is not None:
        cs, ct, cr = _orb_matches(
            gray, src_gray, validation_mask, limit_displacement=limit_displacement
        )
        cs, ct = _refine(gray, src_gray, cs, ct, cr)
        cs, ct = _distinct_points(cs, ct)
        if len(cs) < 6:
            return Registration(
                False,
                "insufficient_additional_validation",
                raw_matches=raw,
                refined_matches=len(source) + len(cs),
            )
        check = np.concatenate((check, np.ones(len(cs), bool)))
        groups = np.concatenate(
            (groups, np.full(len(cs), "additional_mask", dtype=object))
        )
        source, target = np.vstack((source, cs)), np.vstack((target, ct))
        provenance = (
            "spatial_holdout_plus_disjoint_regions_same_features_"
            "not_independent_ground_truth"
        )
    fit = ~check
    if fit.sum() < 25:
        return Registration(
            False,
            "insufficient_refined_features",
            matches=int(fit.sum()),
            raw_matches=raw,
            refined_matches=len(source),
        )
    matrix, selected = cv2.estimateAffinePartial2D(
        source[fit],
        target[fit],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=3000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None or selected is None:
        return Registration(
            False, "background_fit_failed", matches=int(fit.sum()), raw_matches=raw
        )
    good = np.zeros(len(source), bool)
    good[fit] = selected.ravel().astype(bool)
    good[check] = _errors(matrix, source[check], target[check]) <= 2
    return _evaluate(
        source,
        target,
        check,
        matrix,
        good,
        gray.shape,
        method="orb_native_holdout",
        raw_matches=raw,
        provenance=provenance,
        check_groups=groups,
    )

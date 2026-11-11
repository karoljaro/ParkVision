"""Frozen occupancy rules, operating only on saved slot projections."""

from math import isfinite

import cv2
import numpy as np

BASELINE_ID = "occupancy-offline-baseline-v1"


def intersection_area(box, polygon):
    """Preserve the convex baseline and clip arbitrary simple ROIs when needed."""
    x1, y1, x2, y2 = box
    if cv2.isContourConvex(polygon):
        rectangle = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return max(0.0, float(cv2.intersectConvexConvex(rectangle, polygon)[0]))

    # Sutherland-Hodgman clipping against the four half-planes of the bbox.
    # Concave ROIs may create boundary bridges that cancel in the signed area.
    clipped = [tuple(point) for point in polygon.astype(float)]
    for axis, boundary, direction in ((0, x1, 1), (0, x2, -1), (1, y1, 1), (1, y2, -1)):
        if not clipped:
            return 0.0
        result = []
        previous = clipped[-1]
        previous_inside = direction * (previous[axis] - boundary) >= 0
        for current in clipped:
            current_inside = direction * (current[axis] - boundary) >= 0
            if current_inside != previous_inside:
                fraction = (boundary - previous[axis]) / (
                    current[axis] - previous[axis]
                )
                result.append(
                    tuple(a + fraction * (b - a) for a, b in zip(previous, current))
                )
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        clipped = result
    if len(clipped) < 3:
        return 0.0
    following = clipped[1:] + clipped[:1]
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(clipped, following))) / 2


def assign_vehicles(projections, width, height, detections):
    polygons = {
        p.parking_slot_id: np.array(
            [(v.x, v.y) for v in p.polygon.vertices], dtype=np.float32
        )
        * np.float32((width - 1, height - 1))
        for p in projections
    }
    assignments = {slot_id: [] for slot_id in polygons}
    for detection_id, detection in enumerate(detections):
        score = detection["score"]
        box = detection["bbox"]
        if (
            type(score) not in (int, float)
            or not isfinite(score)
            or not 0 <= score <= 1
            or len(box) != 4
            or any(type(v) not in (int, float) or not isfinite(v) for v in box)
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError("Invalid detector output")
        if score < 0.60 or detection["category"] not in {
            "car",
            "bus",
            "truck",
            "motorcycle",
        }:
            continue
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        ranked = sorted(
            (
                (slot_id, intersection_area(box, polygon) / box_area)
                for slot_id, polygon in polygons.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        slot_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best > 0.15 and best - second > 0.08:
            assignments[slot_id].append(detection_id)
    return assignments


def classify(probability, detector_ids):
    if (
        type(probability) not in (float, int)
        or not isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Invalid per-slot score")
    if probability >= 0.90:
        return "occupied", None
    if probability <= 0.10:
        if detector_ids:
            return "unknown", "model_conflict"
        return "free", None
    return "unknown", "low_confidence"

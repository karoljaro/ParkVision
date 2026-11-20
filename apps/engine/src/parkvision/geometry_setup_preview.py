"""Render the reference, line evidence and draft ROI in separate panels."""

from collections import defaultdict
from collections.abc import Callable
from math import hypot
from textwrap import wrap

import cv2
import numpy as np
from numpy.typing import NDArray

from .detection.initialization import GeometryInitialization
from .detection.segment_merging import PixelPoint

type Color = tuple[int, int, int]
_HEADER = 51
_FOOTER = 75
_PANEL_GAP = 10
_BACKGROUND = (25, 26, 30)
_WHITE = (236, 237, 239)
_RAW = (161, 128, 86)
_AXIS = (239, 202, 69)
_OBSERVED = (117, 235, 93)
_ESTIMATED = (46, 166, 255)
_UNRESOLVED = (229, 132, 231)
_PARTIAL = (235, 210, 75)


def _point(xy: PixelPoint) -> tuple[int, int]:
    return round(xy[0]), round(xy[1])


def _line(
    image: NDArray[np.uint8],
    start: PixelPoint,
    end: PixelPoint,
    color: Color,
    width: int,
) -> None:
    cv2.line(image, _point(start), _point(end), color, width, cv2.LINE_AA)


def _draw_dashes(image: NDArray[np.uint8], start: PixelPoint, end: PixelPoint) -> None:
    height, width = image.shape[:2]
    visible, first, last = cv2.clipLine(
        (0, 0, width, height), _point(start), _point(end)
    )
    if not visible:
        return
    dx, dy = last[0] - first[0], last[1] - first[1]
    length = hypot(dx, dy)
    if length <= 1:
        _line(image, first, last, _ESTIMATED, 2)
        return
    for offset in range(0, round(length), 12):
        stop = min(offset + 7, length)
        begin = (first[0] + dx * offset / length, first[1] + dy * offset / length)
        finish = (first[0] + dx * stop / length, first[1] + dy * stop / length)
        _line(image, begin, finish, _ESTIMATED, 2)


def _caption(
    image: NDArray[np.uint8],
    text: str,
    xy: tuple[int, int],
    color: Color = _WHITE,
    scale: float = 0.58,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _badge(
    image: NDArray[np.uint8], text: str, position: PixelPoint, color: Color
) -> None:
    height, width = image.shape[:2]
    x, y = _point(position)
    (label_width, label_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.49, 1
    )
    x = min(max(0, x + 4), max(0, width - label_width - 8))
    y = min(max(0, y + 4), max(0, height - label_height - baseline - 7))
    cv2.rectangle(
        image,
        (x, y),
        (x + label_width + 7, y + label_height + baseline + 6),
        _BACKGROUND,
        -1,
    )
    _caption(image, text, (x + 3, y + label_height + 3), color, 0.49)


def _line_panel(
    image: NDArray[np.uint8],
    result: GeometryInitialization,
    display: Callable[[PixelPoint], PixelPoint],
) -> None:
    for line in result.segments:
        _line(image, display(line.start), display(line.end), _RAW, 1)
    axis_ids = {pair.centerline.segment_id for pair in result.pairs}
    for pair in result.pairs:
        start, end = display(pair.centerline.start), display(pair.centerline.end)
        _line(image, start, end, _BACKGROUND, 4)
        _line(image, start, end, _AXIS, 2)
    used_context = {
        line_id
        for candidate in result.candidates
        if candidate.proposal is not None
        for span in candidate.boundaries
        for line_id in span.line_ids
        if line_id not in axis_ids
    }
    for line in result.topology.lines:
        if line.segment_id in used_context:
            _line(image, display(line.start), display(line.end), _AXIS, 1)
    for endpoint in result.topology.endpoints:
        if endpoint.line_id in axis_ids:
            cv2.circle(
                image, _point(display(endpoint.position)), 3, _AXIS, 1, cv2.LINE_AA
            )


def _topology_panel(
    image: NDArray[np.uint8],
    result: GeometryInitialization,
    display: Callable[[PixelPoint], PixelPoint],
) -> None:
    for pair in result.pairs:
        _line(
            image, display(pair.centerline.start), display(pair.centerline.end), _RAW, 1
        )
    used_nodes = {
        node_id for candidate in result.candidates for node_id in candidate.junction_ids
    }
    for node in result.topology.junctions:
        if not node.intersections:
            continue
        position = display(node.position)
        hit = min(
            node.intersections,
            key=lambda item: (item.hypothetical, item.uses_unpaired_context),
        )
        color = _ESTIMATED if hit.hypothetical else _OBSERVED
        cv2.circle(image, _point(position), 4, _BACKGROUND, 3, cv2.LINE_AA)
        cv2.circle(image, _point(position), 4, color, 1, cv2.LINE_AA)
        if node.candidate_id in used_nodes:
            _badge(image, f"{hit.topology}{node.candidate_id}", position, color)


def _row_panel(
    image: NDArray[np.uint8],
    result: GeometryInitialization,
    display: Callable[[PixelPoint], PixelPoint],
) -> None:
    for row in result.rows:
        if len(row.separator_ids) < 3 and row.geometry_kind == "shared_marking":
            continue
        color = (
            _UNRESOLVED
            if row.rejection_code
            else (_AXIS if row.geometry_kind == "shared_marking" else _PARTIAL)
        )
        for start, end in zip(row.positions, row.positions[1:]):
            _line(image, display(start), display(end), color, 2)
        for index, position in enumerate(row.positions):
            point = _point(display(position))
            kind = row.anchor_kinds[index] if row.anchor_kinds else "paint_junction"
            marker = (
                cv2.MARKER_SQUARE if kind == "boundary_context" else cv2.MARKER_DIAMOND
            )
            cv2.drawMarker(image, point, color, marker, 8, 1, cv2.LINE_AA)
        if row.positions:
            position = tuple(
                sum(p[axis] for p in row.positions) / len(row.positions)
                for axis in (0, 1)
            )
            label = (
                "REJECTED"
                if row.rejection_code
                else ("RAIL" if row.geometry_kind == "shared_marking" else "OPEN")
            )
            _badge(image, f"R{row.row_id} {label}", display(position), color)


def _roi_panel(
    image: NDArray[np.uint8],
    result: GeometryInitialization,
    display: Callable[[PixelPoint], PixelPoint],
) -> None:
    height, width = image.shape[:2]
    junction_positions = {
        node.candidate_id: display(node.position) for node in result.topology.junctions
    }
    for index, candidate in enumerate(result.candidates, start=1):
        if candidate.geometry_kind == "partial" and candidate.rejection_code:
            pixels = tuple(display(p) for p in candidate.anchor_positions)
            for boundary in candidate.boundaries:
                _line(
                    image,
                    display(boundary.start),
                    display(boundary.end),
                    _UNRESOLVED,
                    1,
                )
            for position in pixels:
                cv2.drawMarker(
                    image,
                    _point(position),
                    _UNRESOLVED,
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2,
                    cv2.LINE_AA,
                )
            color, badge = _UNRESOLVED, f"R{index} REJECTED"
        elif candidate.geometry_kind == "partial":
            pixels = tuple(display(p) for p in candidate.anchor_positions)
            for boundary in candidate.boundaries:
                _line(
                    image, display(boundary.start), display(boundary.end), _PARTIAL, 2
                )
            color = _PARTIAL
            badge = f"P{index}"
        elif candidate.proposal is not None:
            pixels = tuple(
                (point.x * (width - 1), point.y * (height - 1))
                for point in candidate.proposal.polygon.vertices
            )
            contour = np.array([_point(xy) for xy in pixels], dtype=np.int32)
            cv2.polylines(image, [contour], True, _BACKGROUND, 4, cv2.LINE_AA)
            cv2.polylines(image, [contour], True, _WHITE, 1, cv2.LINE_AA)
            color = _OBSERVED if candidate.status == "observed" else _ESTIMATED
            badge = f"{index} {'OBS' if candidate.status == 'observed' else 'EST'}"
        else:
            pixels = tuple(display(p) for p in candidate.anchor_positions) or tuple(
                junction_positions[node_id]
                for node_id in candidate.junction_ids
                if node_id in junction_positions
            )
            for position in pixels:
                cv2.drawMarker(
                    image,
                    _point(position),
                    _UNRESOLVED,
                    cv2.MARKER_TILTED_CROSS,
                    12,
                    2,
                    cv2.LINE_AA,
                )
            color = _UNRESOLVED
            badge = f"{index} ?"
        for boundary in (
            () if candidate.geometry_kind == "partial" else candidate.boundaries
        ):
            start, end = display(boundary.start), display(boundary.end)
            if boundary.kind == "estimated":
                _draw_dashes(image, start, end)
            else:
                _line(image, start, end, _OBSERVED, 2)
        if pixels:
            x = sum(x for x, _ in pixels) / len(pixels)
            y = sum(y for _, y in pixels) / len(pixels)
            _badge(image, badge, (x, y), color)


def render_initialization_preview(
    image: NDArray[np.uint8], result: GeometryInitialization
) -> NDArray[np.uint8]:
    """Show each panel at the same display scale while preserving native evidence."""
    native_height, native_width = image.shape[:2]
    partial_count = sum(
        c.geometry_kind == "partial" and c.rejection_code is None
        for c in result.candidates
    )
    rejected_count = sum(c.rejection_code is not None for c in result.candidates)
    scale = min(1.0, 960 / native_width, 720 / native_height)
    width, height = max(2, round(native_width * scale)), max(
        2, round(native_height * scale)
    )
    if scale < 1:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    x_scale = (width - 1) / (native_width - 1)
    y_scale = (height - 1) / (native_height - 1)

    def display(xy: PixelPoint) -> PixelPoint:
        return xy[0] * x_scale, xy[1] * y_scale

    panel_width = max(width, 330)
    rejected = defaultdict(list)
    for index, candidate in enumerate(result.candidates, start=1):
        if candidate.status == "unresolved":
            rejected[candidate.reason].append(str(index))
    rejection_lines = [
        line
        for reason, ids in rejected.items()
        for line in wrap(
            f"{', '.join(ids)}  |  {reason}",
            width=110,
            subsequent_indent="    ",
        )
    ]
    rejection_height = 38 + 22 * len(rejection_lines) if rejection_lines else 0
    row_height = _HEADER + height
    preview = np.empty(
        (
            2 * row_height + _PANEL_GAP + _FOOTER + rejection_height,
            3 * panel_width + 2 * _PANEL_GAP,
            3,
        ),
        dtype=np.uint8,
    )
    preview[:] = _BACKGROUND
    for index, title in enumerate(
        (
            "REFERENCE",
            "LSD / MARKING AXES",
            "L-T-X / ANCHORS",
            "ROW STRUCTURE",
            "FULL ROI / PARTIAL EVIDENCE",
            "MISSING BOUNDARIES",
        )
    ):
        offset = (index % 3) * (panel_width + _PANEL_GAP)
        top = (index // 3) * (row_height + _PANEL_GAP)
        _caption(preview, title, (offset + 13, top + 33), scale=0.65)
        panel = preview[top + _HEADER : top + row_height, offset : offset + width]
        if index < 5:
            panel[:] = image
        if index == 1:
            _line_panel(panel, result, display)
        elif index == 2:
            _topology_panel(panel, result, display)
        elif index == 3:
            _row_panel(panel, result, display)
        elif index == 4:
            _roi_panel(panel, result, display)
        elif index == 5:
            partials = [
                c
                for c in result.candidates
                if c.geometry_kind == "partial" and c.rejection_code is None
            ]
            missing = sorted({kind for c in partials for kind in c.missing_boundaries})
            notes = [f"Partial marking hypotheses: {len(partials)}"]
            notes.append(f"Rejected partials: {rejected_count}")
            notes.extend(f"- {kind.replace('_', ' ')}" for kind in missing)
            notes += [
                "Squares: unclassified boundary context",
                "Diamonds: junction or aligned endpoint",
                "Review the physical extent and boundary type.",
            ]
            y = 22
            for note in notes:
                for line in wrap(note, max(30, int(width / 9))):
                    _caption(panel, line, (12, y), _PARTIAL, 0.50)
                    y += 22
    footer = 2 * row_height + _PANEL_GAP
    labels = (
        ("LSD segments", _RAW),
        ("Paired axis / endpoint", _AXIS),
        ("Solid: observed", _OBSERVED),
        ("Dashed: estimated", _ESTIMATED),
        ("?: unresolved", _UNRESOLVED),
        ("P: partial evidence", _PARTIAL),
    )
    legend_step = (preview.shape[1] - 28) / len(labels)
    for index, (text, color) in enumerate(labels):
        _caption(
            preview, text, (14 + round(index * legend_step), footer + 27), color, 0.50
        )
    observed, estimated, unresolved = (
        sum(
            candidate.status == status and candidate.geometry_kind != "partial"
            for candidate in result.candidates
        )
        for status in ("observed", "estimated", "unresolved")
    )
    summary = (
        f"DRAFT | {observed} observed / {estimated} estimated / {unresolved} unresolved"
        f" / {partial_count} partial / {rejected_count} rejected"
        " | IDs refer to evidence.json"
    )
    _caption(preview, summary, (14, footer + 54), scale=0.59)
    if rejection_lines:
        start = footer + _FOOTER
        _caption(preview, "UNRESOLVED CANDIDATES", (14, start + 20), _UNRESOLVED)
        for index, line in enumerate(rejection_lines, start=1):
            _caption(preview, line, (14, start + 20 + index * 22), scale=0.54)
    return preview

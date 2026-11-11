"""Draw source polygons beside the untouched pixels of the same detail."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from research.vehicle_slots_v41.relations import Shape

TARGET = (50, 238, 255)
CAR = (255, 170, 55)
NEIGHBOR = (191, 164, 255)
FONT = ImageFont.load_default(size=14)


def distance(left: Shape, right: Shape) -> float:
    """Pixel gap between envelopes, used only to display local context."""
    a, b, c, d = left.bounds
    x, y, u, v = right.bounds
    return max(0, x - c, a - u) ** 2 + max(0, y - d, b - v) ** 2


def render(
    image: Image.Image,
    car_id: int,
    target_id: int,
    car: Shape,
    slots: dict[int, Shape],
    owners: tuple[int, ...],
    output: Path,
    prefix: str,
) -> tuple[tuple[int, ...], str, str, str]:
    target = slots[target_id]
    close = sorted(
        (key for key in slots if key != target_id),
        key=lambda key: (distance(slots[key], target), key),
    )
    neighbors = tuple(
        dict.fromkeys([key for key in owners if key != target_id] + close[:3])
    )
    shapes = [(slots[key], NEIGHBOR, f"S{key % 100000:05}") for key in neighbors]
    shapes += [
        (target, TARGET, f"S{target_id % 100000:05}"),
        (car, CAR, f"C{car_id % 100000:05}"),
    ]
    points = [point for shape, _, _ in shapes for p in shape.polygons for point in p]
    x, y = (int(min(point[i] for point in points)) for i in (0, 1))
    u, v = (int(max(point[i] for point in points)) for i in (0, 1))
    bounds = (
        max(0, x - 22),
        max(0, y - 22),
        min(image.width, u + 23),
        min(image.height, v + 23),
    )
    # Upscale the same crop before drawing annotations so thin lines remain legible.
    raw = image.crop(bounds)
    factor = max(1, min(4, 680 / raw.width, 550 / raw.height))
    raw = raw.resize(
        (round(raw.width * factor), round(raw.height * factor)),
        Image.Resampling.LANCZOS,
    )
    paths = []
    for view in ("full-overlay", "detail-original", "detail-overlay"):
        path = output / f"{prefix}-{view}.png"
        paths.append(f"images/{path.name}")
        canvas = image.copy() if view == "full-overlay" else raw.copy()
        offset_x, offset_y = (0, 0) if view == "full-overlay" else bounds[:2]
        scale = 1 if view == "full-overlay" else factor
        if view != "detail-original":
            draw = ImageDraw.Draw(canvas)
            for shape, color, label in shapes:
                for polygon in shape.polygons:
                    coords = [
                        (round((px - offset_x) * scale), round((py - offset_y) * scale))
                        for px, py in polygon
                    ]
                    draw.line(coords + coords[:1], fill=color, width=2)
                ax, ay, bx, by = shape.bounds
                label_x = (ax - offset_x) * scale
                label_y = (ay - offset_y) * scale - 16
                if color == CAR:
                    label_x = ((ax + bx) / 2 - offset_x) * scale
                    label_y = ((ay + by) / 2 - offset_y) * scale
                draw.text(
                    (max(2, label_x), max(2, label_y)),
                    label,
                    fill=color,
                    font=FONT,
                    stroke_width=2,
                    stroke_fill="#181e24",
                )
        with path.open("xb") as stream:
            canvas.save(stream, format="PNG")
    return neighbors, *paths

"""Render original pixels with source slot outlines and car polygons."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .relations import Frame, Link, polygons
from .source import FrameSource

CYAN = (45, 238, 255)
ORANGE = (255, 161, 65)
VIOLET = (239, 138, 255)


def label(draw, position, text, color, font):
    x, y = position
    draw.text(
        (max(4, min(x, 1240)), max(4, min(y, 698))),
        text,
        fill=color,
        font=font,
        stroke_width=2,
        stroke_fill=(12, 19, 22),
    )


def render(
    original: Image.Image,
    frame: Frame,
    spots: FrameSource,
    cars: FrameSource | None,
    output: Path,
) -> None:
    size = original.size
    font = ImageFont.load_default(size=12)
    slot_map = {s.annotation_id: s for s in frame.slots}
    vehicle_map = {v.annotation_id: v for v in frame.vehicles}
    marked = {
        identity
        for slot in frame.slots
        if slot.link == Link.AMBIGUOUS
        for identity in slot.car_annotation_ids
    }
    for view in ("combined", "slots", "cars"):
        layer = Image.new("RGBA", size)
        draw = ImageDraw.Draw(layer)
        if cars and view != "slots":
            for row in cars.annotations:
                ident = row["id"]
                if not vehicle_map[ident].mask_valid:
                    continue
                shape = polygons(row, size)
                color = VIOLET if ident in marked else ORANGE
                for poly in shape.polygons:
                    draw.polygon(
                        poly, fill=(*color, 72), outline=(*color, 235), width=2
                    )
                if view == "cars" or ident in marked:
                    label(draw, shape.bounds[:2], f"C{ident % 100000:05}", color, font)
        if view != "cars":
            for row in spots.annotations:
                slot = slot_map[row["id"]]
                if slot.camera_slot_key is None:
                    continue
                shape = polygons(row, size)
                color = VIOLET if slot.link == Link.AMBIGUOUS else CYAN
                for poly in shape.polygons:
                    draw.line((*poly, poly[0]), fill=(*color, 245), width=2)
                label(draw, shape.bounds[:2], f"S{row['id'] % 100000:05}", color, font)
        merged = Image.alpha_composite(original.convert("RGBA"), layer).convert("RGB")
        page = Image.new("RGB", (size[0], size[1] + 39), (21, 32, 37))
        page.paste(merged, (0, 39))
        title = ImageDraw.Draw(page)
        desc = f"UFPR04 {frame.source_capture_time} | {frame.image_id} | {view}"
        title.text((12, 5), desc, fill=(255, 255, 255), font=font)
        legend = "S: source spot | C: car annotation | violet: ambiguous car_id"
        if not frame.cars_frame_present:
            legend += " | cars frame absent in source"
        title.text((12, 20), legend, fill=(183, 209, 218), font=font)
        with (output / f"{frame.image_id}-{view}.png").open("xb") as stream:
            page.save(stream, format="PNG")

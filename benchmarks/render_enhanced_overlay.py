#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (255, 99, 132),
    (54, 162, 235),
    (255, 206, 86),
    (75, 192, 192),
    (153, 102, 255),
    (255, 159, 64),
    (0, 200, 83),
    (255, 64, 129),
]

FONT_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

FONT_REGULAR_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


@dataclass
class ZoneRow:
    index: int
    well: str
    center_x: float
    center_y: float
    diameter_eq_mm: float
    area_mm2: float
    confidence: float
    mask_path: Path


@dataclass
class LabelSpec:
    zone: ZoneRow
    color: tuple[int, int, int]
    lines: list[str]
    box_w: int
    box_h: int
    x: float
    y: float
    preferred_y: float
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a manuscript-grade overlay with clearer callouts.")
    parser.add_argument("--photo", required=True, help="Path to the original plate photo.")
    parser.add_argument("--phenotypes-csv", required=True, help="Path to well_phenotypes.csv.")
    parser.add_argument("--plate-id", required=True, help="Plate id, usually the image stem.")
    parser.add_argument("--zonevision-root", required=True, help="Path to algorithms/ZoneVision.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--alpha", type=float, default=0.24, help="Mask fill alpha.")
    return parser.parse_args()


def pick_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def alpha_fill(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    output[mask] = (output[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return output


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    output = mask.copy()
    for _ in range(iterations):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        neighbors = []
        for dy in range(3):
            for dx in range(3):
                neighbors.append(padded[dy:dy + output.shape[0], dx:dx + output.shape[1]])
        output = np.logical_or.reduce(neighbors)
    return output


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            interior &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return np.logical_and(mask, np.logical_not(interior))


def load_zone_rows(phenotypes_csv: Path, plate_id: str, zonevision_root: Path) -> list[ZoneRow]:
    rows: list[ZoneRow] = []
    with phenotypes_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw.get("plate_id") != plate_id or raw.get("has_zone") != "True":
                continue
            mask_value = (raw.get("mask_path") or "").replace("\\", "/")
            mask_path = Path(mask_value)
            if not mask_path.is_absolute():
                mask_path = zonevision_root / mask_path
            rows.append(
                ZoneRow(
                    index=len(rows) + 1,
                    well=raw["well"],
                    center_x=float(raw["center_x_px"]),
                    center_y=float(raw["center_y_px"]),
                    diameter_eq_mm=float(raw["diameter_eq_mm"]),
                    area_mm2=float(raw["area_mm2"]),
                    confidence=float(raw["confidence"]),
                    mask_path=mask_path,
                )
            )
    if not rows:
        raise SystemExit(f"No zone rows found for plate_id={plate_id} in {phenotypes_csv}")
    return rows


def label_group(center_x: float, center_y: float, image_w: int, image_h: int) -> str:
    return ("l" if center_x < image_w / 2 else "r") + ("t" if center_y < image_h / 2 else "b")


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        return value
    return max(low, min(high, value))


def measure_multiline(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    well_font: ImageFont.ImageFont,
    metric_font: ImageFont.ImageFont,
    line_gap: int,
) -> tuple[int, int]:
    widths: list[int] = []
    heights: list[int] = []
    for idx, line in enumerate(lines):
        font = well_font if idx == 0 else metric_font
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        widths.append(right - left)
        heights.append(bottom - top)
    return max(widths), sum(heights) + line_gap * (len(lines) - 1)


def build_label_specs(
    zones: list[ZoneRow],
    image_w: int,
    image_h: int,
    draw: ImageDraw.ImageDraw,
    well_font: ImageFont.ImageFont,
    metric_font: ImageFont.ImageFont,
) -> list[LabelSpec]:
    specs: list[LabelSpec] = []
    image_cx = image_w / 2
    image_cy = image_h / 2
    box_pad_x = 20
    box_pad_y = 15
    line_gap = 5
    radial_offset = max(132.0, min(image_w, image_h) * 0.095)
    margin = 22.0
    half_gap = 26.0

    for zone in zones:
        color = PALETTE[(zone.index - 1) % len(PALETTE)]
        lines = [
            zone.well,
            f"D = {zone.diameter_eq_mm:.1f} mm",
            f"Area = {zone.area_mm2:.1f} mm^2",
        ]
        text_w, text_h = measure_multiline(draw, lines, well_font, metric_font, line_gap)
        box_w = text_w + box_pad_x * 2
        box_h = text_h + box_pad_y * 2

        vx = zone.center_x - image_cx
        vy = zone.center_y - image_cy
        norm = math.hypot(vx, vy) or 1.0
        ux = vx / norm
        uy = vy / norm
        preferred_center_x = zone.center_x + ux * radial_offset
        preferred_center_y = zone.center_y + uy * radial_offset
        group = label_group(zone.center_x, zone.center_y, image_w, image_h)

        if group.startswith("l"):
            min_x = margin
            max_x = image_cx - half_gap - box_w
        else:
            min_x = image_cx + half_gap
            max_x = image_w - margin - box_w
        if group.endswith("t"):
            min_y = margin
            max_y = image_cy - half_gap - box_h
        else:
            min_y = image_cy + half_gap
            max_y = image_h - margin - box_h

        x = clamp(preferred_center_x - box_w / 2, min_x, max_x)
        y = clamp(preferred_center_y - box_h / 2, min_y, max_y)
        specs.append(
            LabelSpec(
                zone=zone,
                color=color,
                lines=lines,
                box_w=box_w,
                box_h=box_h,
                x=x,
                y=y,
                preferred_y=y,
                group=group,
            )
        )

    for group in ("lt", "lb", "rt", "rb"):
        group_specs = [spec for spec in specs if spec.group == group]
        if not group_specs:
            continue
        min_y = 18.0 if group.endswith("t") else image_h / 2 + 22.0
        max_y = image_h / 2 - 22.0 if group.endswith("t") else image_h - 18.0
        group_specs.sort(key=lambda spec: spec.preferred_y)
        cursor = min_y
        gap = 10.0
        for spec in group_specs:
            spec.y = max(spec.preferred_y, cursor)
            cursor = spec.y + spec.box_h + gap
        overflow = cursor - gap - max_y
        if overflow > 0:
            for spec in reversed(group_specs):
                spec.y -= overflow
                if spec.y < min_y:
                    overflow = min_y - spec.y
                    spec.y = min_y
                else:
                    overflow = 0.0
                if overflow <= 0:
                    break
        for idx in range(1, len(group_specs)):
            prev = group_specs[idx - 1]
            curr = group_specs[idx]
            if curr.y < prev.y + prev.box_h + gap:
                curr.y = prev.y + prev.box_h + gap
        for spec in group_specs:
            upper = max_y - spec.box_h
            lower = min_y
            spec.y = clamp(spec.y, lower, upper)
    return specs


def nearest_point_on_box(px: float, py: float, box_x: float, box_y: float, box_w: int, box_h: int) -> tuple[float, float]:
    return clamp(px, box_x, box_x + box_w), clamp(py, box_y, box_y + box_h)


def main() -> None:
    args = parse_args()
    zonevision_root = Path(args.zonevision_root).resolve()
    photo_path = Path(args.photo).resolve()
    phenotypes_csv = Path(args.phenotypes_csv).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zones = load_zone_rows(phenotypes_csv, args.plate_id, zonevision_root)

    base_image = np.asarray(Image.open(photo_path).convert("RGB"))
    overlay = base_image.copy()

    for zone in zones:
        if not zone.mask_path.exists():
            raise SystemExit(f"Missing mask: {zone.mask_path}")
        mask_bool = np.asarray(Image.open(zone.mask_path).convert("L")) > 0
        color = PALETTE[(zone.index - 1) % len(PALETTE)]
        overlay = alpha_fill(overlay, mask_bool, color, args.alpha)
        edge = boundary_mask(mask_bool)
        edge_outer = dilate_mask(edge, iterations=2)
        edge_inner = dilate_mask(edge, iterations=1)
        overlay[edge_outer] = (255, 255, 255)
        overlay[edge_inner] = color

    canvas = Image.fromarray(overlay).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    image_w, image_h = canvas.size
    well_font = pick_font(FONT_BOLD_CANDIDATES, max(30, round(min(image_w, image_h) * 0.0172)))
    metric_font = pick_font(FONT_REGULAR_CANDIDATES, max(28, round(min(image_w, image_h) * 0.0156)))
    specs = build_label_specs(zones, image_w, image_h, draw, well_font, metric_font)

    box_bg = (255, 255, 255, 238)
    text_color = (28, 28, 28, 255)
    line_gap = 5
    text_pad_x = 20
    text_pad_y = 15

    for spec in specs:
        box = [spec.x, spec.y, spec.x + spec.box_w, spec.y + spec.box_h]
        start_x, start_y = spec.zone.center_x, spec.zone.center_y
        end_x, end_y = nearest_point_on_box(start_x, start_y, spec.x, spec.y, spec.box_w, spec.box_h)
        draw.line((start_x, start_y, end_x, end_y), fill=(255, 255, 255, 220), width=7)
        draw.line((start_x, start_y, end_x, end_y), fill=spec.color + (245,), width=3)
        draw.rounded_rectangle(box, radius=12, fill=box_bg, outline=spec.color + (255,), width=3)

        text_x = spec.x + text_pad_x
        text_y = spec.y + text_pad_y
        for idx, line in enumerate(spec.lines):
            font = well_font if idx == 0 else metric_font
            fill = spec.color + (255,) if idx == 0 else text_color
            draw.text((text_x, text_y), line, fill=fill, font=font)
            left, top, right, bottom = draw.textbbox((text_x, text_y), line, font=font)
            text_y += (bottom - top) + line_gap

    canvas.convert("RGB").save(output_path, dpi=(300, 300))
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()

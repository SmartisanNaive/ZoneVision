from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import ZoneVisionConfig
from .io_utils import ensure_dir, write_csv
from .plate import PlateGeometry


MeasureFn = Callable[[np.ndarray, float | None], dict[str, float | int | None]]


def write_well_phenotype_files(output_dir: str | Path, rows: list[dict[str, object]]) -> None:
    output = ensure_dir(output_dir)
    write_csv(output / "well_phenotypes.csv", rows)
    (output / "well_phenotypes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def build_well_phenotype_rows(
    image_path: Path,
    detections: list,
    geometry: PlateGeometry,
    measurement_rows: list[dict[str, object]],
    config: ZoneVisionConfig,
    measure_mask: MeasureFn,
) -> list[dict[str, object]]:
    row_centers, col_centers, grid_flags = complete_grid(geometry)
    measurements_by_instance = {int(row["instance_id"]): row for row in measurement_rows if row.get("image_name") == image_path.name}
    best_by_well: dict[tuple[int, int], tuple[int, object]] = {}

    for index, detection in enumerate(detections, start=1):
        if detection.row is None or detection.col is None:
            continue
        if detection.mask is None or detection.mask.sum() == 0:
            continue
        key = (int(detection.row), int(detection.col))
        current = best_by_well.get(key)
        if current is None or float(detection.confidence) > float(current[1].confidence):
            best_by_well[key] = (index, detection)

    rows: list[dict[str, object]] = []
    for row_index in range(8):
        for col_index in range(12):
            well = f"{chr(ord('A') + row_index)}{col_index + 1}"
            well_center_x = col_centers[col_index] if col_index < len(col_centers) else None
            well_center_y = row_centers[row_index] if row_index < len(row_centers) else None
            selected = best_by_well.get((row_index, col_index))
            if selected is None:
                rows.append(_empty_well_row(image_path, well, row_index, col_index, well_center_x, well_center_y, grid_flags))
                continue

            instance_id, detection = selected
            metrics = measure_mask(detection.mask, geometry.px_per_mm)
            measurement = measurements_by_instance.get(instance_id, {})
            center_offset_mm = None
            if well_center_x is not None and well_center_y is not None and geometry.px_per_mm:
                center_offset_px = float(np.hypot(float(detection.center_x) - well_center_x, float(detection.center_y) - well_center_y))
                center_offset_mm = center_offset_px / geometry.px_per_mm
            qc_flags = phenotype_qc_flags(detection.mask, metrics, center_offset_mm, config, geometry)
            qc_flags.extend(grid_flags)
            rows.append(
                {
                    "image_name": image_path.name,
                    "plate_id": image_path.stem,
                    "well": well,
                    "row": chr(ord("A") + row_index),
                    "col": col_index + 1,
                    "has_zone": True,
                    "diameter_eq_mm": _round_or_none(metrics["diameter_eq_mm"], 3),
                    "diameter_max_mm": _round_or_none(metrics["diameter_max_mm"], 3),
                    "area_mm2": _area_mm2(metrics["area_px"], geometry.px_per_mm),
                    "center_x_px": round(float(detection.center_x), 2),
                    "center_y_px": round(float(detection.center_y), 2),
                    "well_center_x_px": _round_or_none(well_center_x, 2),
                    "well_center_y_px": _round_or_none(well_center_y, 2),
                    "center_offset_mm": _round_or_none(center_offset_mm, 3),
                    "confidence": round(float(detection.confidence), 4),
                    "source": detection.source,
                    "mask_path": measurement.get("mask_path"),
                    "overlay_path": measurement.get("overlay_path"),
                    "qc_flags": ";".join(qc_flags),
                    "qc_status": "review" if qc_flags else "pass",
                }
            )
    return rows


def complete_grid(geometry: PlateGeometry) -> tuple[list[float | None], list[float | None], list[str]]:
    flags: list[str] = []
    row_centers = _complete_axis(geometry.row_centers, 8, geometry.pitch_px)
    col_centers = _complete_axis(geometry.col_centers, 12, geometry.pitch_px)
    if len(geometry.row_centers) < 8 or len(geometry.col_centers) < 12:
        flags.append("incomplete_grid")
    if any(value is None for value in row_centers) or any(value is None for value in col_centers):
        flags.append("grid_center_missing")
    return row_centers, col_centers, flags


def phenotype_qc_flags(
    mask: np.ndarray,
    metrics: dict[str, float | int | None],
    center_offset_mm: float | None,
    config: ZoneVisionConfig,
    geometry: PlateGeometry,
) -> list[str]:
    flags: list[str] = []
    diameter = metrics.get("diameter_eq_mm")
    if diameter is None:
        flags.append("missing_scale")
    else:
        if float(diameter) < config.min_zone_radius_mm * 2.0:
            flags.append("too_small")
        if float(diameter) > config.max_zone_radius_mm * 2.0:
            flags.append("too_large")
    if center_offset_mm is not None and center_offset_mm > config.well_pitch_mm * 0.35:
        flags.append("off_center")
    if _touches_border(mask):
        flags.append("touches_border")
    circularity = _circularity(mask)
    if circularity is not None and circularity < 0.45:
        flags.append("low_circularity")
    if geometry.px_per_mm is None:
        flags.append("geometry_scale_missing")
    return flags


def _complete_axis(values: list[float], expected: int, pitch: float | None) -> list[float | None]:
    if len(values) >= expected:
        return [float(value) for value in sorted(values)[:expected]]
    if not values:
        return [None] * expected
    sorted_values = sorted(float(value) for value in values)
    step = pitch or (float(np.median(np.diff(sorted_values))) if len(sorted_values) > 1 else None)
    if step is None or step <= 0:
        return sorted_values + [None] * (expected - len(sorted_values))
    start = sorted_values[0]
    return [start + index * step for index in range(expected)]


def _empty_well_row(
    image_path: Path,
    well: str,
    row_index: int,
    col_index: int,
    well_center_x: float | None,
    well_center_y: float | None,
    grid_flags: list[str],
) -> dict[str, object]:
    return {
        "image_name": image_path.name,
        "plate_id": image_path.stem,
        "well": well,
        "row": chr(ord("A") + row_index),
        "col": col_index + 1,
        "has_zone": False,
        "diameter_eq_mm": None,
        "diameter_max_mm": None,
        "area_mm2": None,
        "center_x_px": None,
        "center_y_px": None,
        "well_center_x_px": _round_or_none(well_center_x, 2),
        "well_center_y_px": _round_or_none(well_center_y, 2),
        "center_offset_mm": None,
        "confidence": None,
        "source": None,
        "mask_path": None,
        "overlay_path": None,
        "qc_flags": ";".join(["no_zone", *grid_flags]),
        "qc_status": "review" if grid_flags else "pass",
    }


def _area_mm2(area_px: float | int | None, px_per_mm: float | None) -> float | None:
    if area_px is None or not px_per_mm:
        return None
    return round(float(area_px) / (px_per_mm * px_per_mm), 3)


def _round_or_none(value: float | int | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _circularity(mask: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return None
    return float(4.0 * np.pi * area / (perimeter * perimeter))

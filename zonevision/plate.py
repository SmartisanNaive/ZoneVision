from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class WellPoint:
    x: float
    y: float
    radius: float
    row: int | None = None
    col: int | None = None


@dataclass(slots=True)
class PlateGeometry:
    wells: list[WellPoint]
    pitch_px: float | None
    px_per_mm: float | None
    well_radius_px: float | None
    row_centers: list[float]
    col_centers: list[float]


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_plate_geometry(image: np.ndarray, well_pitch_mm: float) -> PlateGeometry:
    gray = to_gray(image)
    blur = cv2.medianBlur(gray, 5)
    min_side = min(gray.shape[:2])
    min_radius = max(10, min_side // 90)
    max_radius = max(min_radius + 4, min_side // 28)
    circles = None

    for param2 in (30, 24, 20, 16):
        found = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min_radius * 2),
            param1=120,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if found is not None and len(found[0]) >= 24:
            circles = found[0]
            break

    if circles is None:
        return PlateGeometry([], None, None, None, [], [])

    unique = _dedupe_circles(np.asarray(circles))
    radii = unique[:, 2]
    well_radius = float(np.median(radii))
    tolerance = max(8.0, well_radius * 1.5)
    row_centers = _merge_axis(unique[:, 1], tolerance, expected=8)
    col_centers = _merge_axis(unique[:, 0], tolerance, expected=12)

    wells: list[WellPoint] = []
    for x, y, radius in unique:
        row = _nearest_cluster(y, row_centers)
        col = _nearest_cluster(x, col_centers)
        wells.append(WellPoint(float(x), float(y), float(radius), row=row, col=col))

    pitch_candidates: list[float] = []
    if len(row_centers) > 1:
        pitch_candidates.extend(np.diff(sorted(row_centers)).tolist())
    if len(col_centers) > 1:
        pitch_candidates.extend(np.diff(sorted(col_centers)).tolist())

    pitch_px = float(np.median(pitch_candidates)) if pitch_candidates else None
    px_per_mm = (pitch_px / well_pitch_mm) if pitch_px else None
    return PlateGeometry(wells, pitch_px, px_per_mm, well_radius, row_centers, col_centers)


def _dedupe_circles(circles: np.ndarray) -> np.ndarray:
    kept: list[np.ndarray] = []
    for circle in sorted(circles, key=lambda item: float(item[2]), reverse=True):
        if not kept:
            kept.append(circle)
            continue
        x, y, radius = circle
        too_close = False
        for existing in kept:
            dist = np.hypot(x - existing[0], y - existing[1])
            if dist < max(radius, existing[2]) * 0.7:
                too_close = True
                break
        if not too_close:
            kept.append(circle)
    return np.asarray(sorted(kept, key=lambda item: (item[1], item[0])))


def _merge_axis(values: np.ndarray, tolerance: float, expected: int) -> list[float]:
    sorted_values = sorted(float(value) for value in values)
    best = _cluster_values(sorted_values, tolerance)
    while len(best) > expected and tolerance < 120:
        tolerance *= 1.15
        best = _cluster_values(sorted_values, tolerance)
    return best


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def _nearest_cluster(value: float, clusters: list[float]) -> int | None:
    if not clusters:
        return None
    distances = [abs(value - cluster) for cluster in clusters]
    return int(np.argmin(distances))

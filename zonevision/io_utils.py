from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def discover_images(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    suffix = target.suffix.lower() or ".png"
    ext = ".png" if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {target}")
    target.write_bytes(encoded.tobytes())


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    write_image(path, (mask.astype(np.uint8) * 255))


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import ensure_dir, read_image


ROBOFLOW_COCO_NAME = "_annotations.coco.json"


@dataclass(slots=True)
class CocoInstance:
    image_id: int | str
    image_file: str
    image_path: Path | None
    width: int
    height: int
    annotation_id: int | str
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]
    segmentation: Any
    area: float
    split: str
    source_json: Path


CocoIndex = dict[str, list[CocoInstance]]


def find_coco_annotation_files(dataset: str | Path) -> list[Path]:
    """Return COCO JSON files from either a single JSON path or a dataset root."""
    root = Path(dataset)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"COCO dataset not found: {root}")

    files = sorted(root.rglob(ROBOFLOW_COCO_NAME))
    files.extend(sorted(root.rglob("instances_*.json")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_coco_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def split_from_annotation_path(path: Path) -> str:
    if path.name == ROBOFLOW_COCO_NAME:
        return path.parent.name
    stem = path.stem
    if stem.startswith("instances_"):
        return stem.replace("instances_", "", 1)
    return path.parent.name


def yolo_split_name(split: str) -> str:
    normalized = split.lower()
    if normalized in {"valid", "validation", "val"}:
        return "val"
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"test", "testing"}:
        return "test"
    return normalized


def find_image_path(annotation_path: Path, file_name: str) -> Path | None:
    candidates = [
        annotation_path.parent / file_name,
        annotation_path.parent / Path(file_name).name,
        annotation_path.parent.parent / file_name,
        annotation_path.parent.parent / annotation_path.parent.name / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_coco_instances(
    dataset: str | Path,
    *,
    category_ids: set[int] | None = None,
    min_area_px: float = 0.0,
) -> list[CocoInstance]:
    instances: list[CocoInstance] = []
    for annotation_path in find_coco_annotation_files(dataset):
        data = load_coco_json(annotation_path)
        split = split_from_annotation_path(annotation_path)
        categories = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
        images = {image["id"]: image for image in data.get("images", [])}

        for annotation in data.get("annotations", []):
            category_id = int(annotation.get("category_id", -1))
            if category_ids is not None and category_id not in category_ids:
                continue
            area = float(annotation.get("area", 0.0) or 0.0)
            if area < min_area_px:
                continue
            image = images.get(annotation.get("image_id"))
            if image is None:
                continue
            bbox_raw = annotation.get("bbox") or [0, 0, 0, 0]
            if len(bbox_raw) != 4:
                bbox_raw = [0, 0, 0, 0]
            file_name = str(image.get("file_name", ""))
            instances.append(
                CocoInstance(
                    image_id=image.get("id"),
                    image_file=file_name,
                    image_path=find_image_path(annotation_path, file_name),
                    width=int(image.get("width") or 0),
                    height=int(image.get("height") or 0),
                    annotation_id=annotation.get("id"),
                    category_id=category_id,
                    category_name=categories.get(category_id, str(category_id)),
                    bbox=tuple(float(v) for v in bbox_raw),
                    segmentation=annotation.get("segmentation"),
                    area=area,
                    split=split,
                    source_json=annotation_path,
                )
            )
    return instances


def build_coco_index(
    dataset: str | Path,
    *,
    category_ids: set[int] | None = None,
    min_area_px: float = 0.0,
) -> CocoIndex:
    index: CocoIndex = defaultdict(list)
    for instance in iter_coco_instances(dataset, category_ids=category_ids, min_area_px=min_area_px):
        keys = {
            instance.image_file,
            Path(instance.image_file).name,
            Path(instance.image_file).stem,
        }
        if instance.image_path is not None:
            keys.add(instance.image_path.name)
            keys.add(instance.image_path.stem)
        for key in keys:
            if key:
                index[key].append(instance)
    return dict(index)


def lookup_instances(index: CocoIndex, image_path: str | Path) -> list[CocoInstance]:
    path = Path(image_path)
    for key in (path.name, path.stem, path.as_posix()):
        if key in index:
            return index[key]
    return []


def segmentation_to_mask(
    segmentation: Any,
    target_shape: tuple[int, int],
    *,
    source_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Rasterize COCO polygon/RLE segmentation to a boolean mask.

    source_size is (width, height) from COCO. If it differs from the image being
    processed, polygon coordinates are scaled into target_shape.
    """
    height, width = target_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        source_width, source_height = source_size or (width, height)
        sx = width / float(source_width or width)
        sy = height / float(source_height or height)
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= sx
            points[:, 1] *= sy
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool) if mask.any() else None

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except Exception:
            return None
        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        if decoded.shape[:2] != (height, width):
            decoded = cv2.resize(decoded.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return decoded.astype(bool) if decoded.any() else None

    return None


def scale_bbox_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    target_shape: tuple[int, int],
    *,
    source_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    height, width = target_shape
    source_width, source_height = source_size or (width, height)
    sx = width / float(source_width or width)
    sy = height / float(source_height or height)
    x, y, w, h = bbox_xywh
    x1 = int(round(x * sx))
    y1 = int(round(y * sy))
    x2 = int(round((x + w) * sx))
    y2 = int(round((y + h) * sy))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def inspect_coco_dataset(dataset: str | Path) -> dict[str, Any]:
    annotation_files = find_coco_annotation_files(dataset)
    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "annotation_files": [str(path) for path in annotation_files],
        "totals": {"images": 0, "annotations": 0, "missing_images": 0},
        "categories": {},
        "splits": {},
        "warnings": [],
    }

    all_category_names: dict[str, set[int]] = defaultdict(set)
    total_images = 0
    total_annotations = 0
    total_missing = 0

    for annotation_path in annotation_files:
        data = load_coco_json(annotation_path)
        split = split_from_annotation_path(annotation_path)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
        for cat_id, cat_name in categories.items():
            all_category_names[str(cat_name)].add(cat_id)
            summary["categories"][str(cat_id)] = cat_name

        image_by_id = {image["id"]: image for image in images}
        anns_per_image = Counter(annotation.get("image_id") for annotation in annotations)
        areas = [float(annotation.get("area", 0.0) or 0.0) for annotation in annotations]
        bbox_widths = [float((annotation.get("bbox") or [0, 0, 0, 0])[2]) for annotation in annotations]
        bbox_heights = [float((annotation.get("bbox") or [0, 0, 0, 0])[3]) for annotation in annotations]
        polygon_points: list[int] = []
        category_counts = Counter(int(annotation.get("category_id", -1)) for annotation in annotations)
        missing_images = 0
        for image in images:
            if find_image_path(annotation_path, str(image.get("file_name", ""))) is None:
                missing_images += 1

        for annotation in annotations:
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if isinstance(polygon, list):
                        polygon_points.append(len(polygon) // 2)

        split_summary = {
            "json": str(annotation_path),
            "images": len(images),
            "annotations": len(annotations),
            "missing_images": missing_images,
            "annotations_per_image": {
                str(image.get("file_name")): int(anns_per_image.get(image.get("id"), 0)) for image in images
            },
            "category_counts": {str(k): int(v) for k, v in sorted(category_counts.items())},
            "area_px": _range_stats(areas),
            "bbox_width_px": _range_stats(bbox_widths),
            "bbox_height_px": _range_stats(bbox_heights),
            "polygon_points": _range_stats(polygon_points),
            "image_size": sorted(
                {f"{int(image.get('width') or 0)}x{int(image.get('height') or 0)}" for image in images}
            ),
        }
        # Keep split names unique if several files map to the same split.
        split_key = split
        suffix = 2
        while split_key in summary["splits"]:
            split_key = f"{split}_{suffix}"
            suffix += 1
        summary["splits"][split_key] = split_summary

        total_images += len(images)
        total_annotations += len(annotations)
        total_missing += missing_images

        orphan_annotations = sum(1 for annotation in annotations if annotation.get("image_id") not in image_by_id)
        if orphan_annotations:
            summary["warnings"].append(f"{annotation_path}: {orphan_annotations} annotations reference missing image ids")

    duplicate_names = {name: sorted(ids) for name, ids in all_category_names.items() if len(ids) > 1}
    if duplicate_names:
        summary["warnings"].append(f"duplicate category names with different ids: {duplicate_names}")

    summary["totals"] = {
        "images": total_images,
        "annotations": total_annotations,
        "missing_images": total_missing,
    }
    return summary


def normalize_coco_dataset(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    class_name: str = "inhibition_zone",
    category_id: int = 1,
    copy_images: bool = True,
    train_ratio: float | None = None,
    valid_ratio: float | None = None,
    test_ratio: float | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    summary: dict[str, Any] = {
        "input": str(dataset),
        "output": str(output),
        "class_name": class_name,
        "category_id": category_id,
        "splits": {},
        "missing_images": [],
        "empty_segmentations": 0,
        "recomputed_boxes": 0,
        "recomputed_areas": 0,
    }

    normalized_category = {"id": category_id, "name": class_name, "supercategory": "zone"}
    records = _collect_coco_records(dataset, summary)
    if train_ratio is not None or valid_ratio is not None or test_ratio is not None:
        records = _resplit_records(records, train_ratio or 0.7, valid_ratio or 0.15, test_ratio or 0.15, seed)

    for split in sorted(records):
        split_dir = ensure_dir(output / split)
        split_records = records[split]
        kept_images: list[dict[str, Any]] = []
        normalized_annotations: list[dict[str, Any]] = []
        copied_images = 0
        dropped_annotations = 0

        for record in split_records:
            image = dict(record["image"])
            source_image = record["image_path"]
            file_name = Path(str(image.get("file_name", ""))).name
            image["file_name"] = file_name
            kept_images.append(image)
            if copy_images:
                shutil.copy2(source_image, split_dir / file_name)
                copied_images += 1

            for annotation in record["annotations"]:
                normalized = _normalize_annotation(annotation, image, category_id, summary)
                if normalized is None:
                    dropped_annotations += 1
                    continue
                normalized_annotations.append(normalized)

        normalized_data = {
            "info": {"description": f"ZoneVision normalized COCO dataset from {dataset}"},
            "licenses": [],
            "categories": [normalized_category],
            "images": kept_images,
            "annotations": normalized_annotations,
        }
        target_json = split_dir / ROBOFLOW_COCO_NAME
        target_json.write_text(json.dumps(normalized_data, ensure_ascii=False, indent=2) + "\n")
        summary["splits"][split] = {
            "json": str(target_json),
            "images": len(kept_images),
            "copied_images": copied_images,
            "annotations": len(normalized_annotations),
            "dropped_annotations": dropped_annotations,
        }

    return summary



def _collect_coco_records(dataset: str | Path, summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_image_ids: set[tuple[str, Any]] = set()
    next_image_id = 1
    next_annotation_id = 1

    for annotation_path in find_coco_annotation_files(dataset):
        data = load_coco_json(annotation_path)
        split = split_from_annotation_path(annotation_path)
        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotation = dict(annotation)
            annotation["id"] = next_annotation_id
            next_annotation_id += 1
            annotations_by_image[annotation.get("image_id")].append(annotation)

        for image in data.get("images", []):
            source_file_name = str(image.get("file_name", ""))
            source_image = find_image_path(annotation_path, source_file_name)
            if source_image is None:
                summary["missing_images"].append(str(annotation_path.parent / source_file_name))
                continue
            source_key = (source_image.resolve().as_posix(), image.get("id"))
            if source_key in used_image_ids:
                continue
            used_image_ids.add(source_key)
            normalized_image = dict(image)
            old_image_id = image.get("id")
            normalized_image["id"] = next_image_id
            next_image_id += 1
            for annotation in annotations_by_image.get(old_image_id, []):
                annotation["image_id"] = normalized_image["id"]
            records[split].append(
                {
                    "image": normalized_image,
                    "image_path": source_image,
                    "annotations": annotations_by_image.get(old_image_id, []),
                }
            )
    return dict(records)


def _resplit_records(
    records: dict[str, list[dict[str, Any]]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio /= total_ratio
    valid_ratio /= total_ratio

    all_records = [record for split_records in records.values() for record in split_records]
    random.Random(seed).shuffle(all_records)
    total = len(all_records)
    train_count = int(round(total * train_ratio))
    valid_count = int(round(total * valid_ratio))
    if total >= 3:
        train_count = max(1, min(train_count, total - 2))
        valid_count = max(1, min(valid_count, total - train_count - 1))
    test_count = total - train_count - valid_count

    return {
        "train": all_records[:train_count],
        "valid": all_records[train_count : train_count + valid_count],
        "test": all_records[train_count + valid_count : train_count + valid_count + test_count],
    }


def _normalize_annotation(
    annotation: dict[str, Any],
    image: dict[str, Any],
    category_id: int,
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    normalized = dict(annotation)
    normalized["category_id"] = category_id
    normalized["iscrowd"] = int(normalized.get("iscrowd", 0) or 0)
    segmentation = normalized.get("segmentation")
    if not segmentation:
        summary["empty_segmentations"] += 1
        return None

    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    bbox = _bbox_from_segmentation(segmentation, width, height)
    area = _area_from_segmentation(segmentation, width, height)
    if bbox is not None:
        normalized["bbox"] = [round(value, 2) for value in bbox]
        summary["recomputed_boxes"] += 1
    if area is not None:
        normalized["area"] = round(float(area), 2)
        summary["recomputed_areas"] += 1
    return normalized


def _bbox_from_segmentation(segmentation: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    points = _segmentation_points(segmentation)
    if points.size == 0:
        return None
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)


def _area_from_segmentation(segmentation: Any, width: int, height: int) -> float | None:
    if width <= 0 or height <= 0:
        return None
    mask = segmentation_to_mask(segmentation, (height, width), source_size=(width, height))
    if mask is None:
        return None
    return float(mask.sum())


def _segmentation_points(segmentation: Any) -> np.ndarray:
    polygons = segmentation if isinstance(segmentation, list) else []
    points: list[np.ndarray] = []
    for polygon in polygons:
        if isinstance(polygon, list) and len(polygon) >= 6:
            points.append(np.asarray(polygon, dtype=np.float32).reshape(-1, 2))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(points)



def export_yolo_seg_dataset(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    class_name: str = "inhibition_zone",
    include_test: bool = True,
    copy_images: bool = True,
) -> dict[str, Any]:
    """Convert Roboflow/COCO polygon segmentation data to Ultralytics YOLO-seg."""
    output = ensure_dir(output_dir)
    images_root = ensure_dir(output / "images")
    labels_root = ensure_dir(output / "labels")
    summary: dict[str, Any] = {
        "output": str(output),
        "yaml": str(output / "data.yaml"),
        "splits": {},
        "skipped_annotations": 0,
        "missing_images": [],
    }

    present_splits: set[str] = set()
    for annotation_path in find_coco_annotation_files(dataset):
        data = load_coco_json(annotation_path)
        source_split = split_from_annotation_path(annotation_path)
        split = yolo_split_name(source_split)
        if split == "test" and not include_test:
            continue
        present_splits.add(split)
        image_dir = ensure_dir(images_root / split)
        label_dir = ensure_dir(labels_root / split)

        images = {image["id"]: image for image in data.get("images", [])}
        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[annotation.get("image_id")].append(annotation)

        split_copied = 0
        split_labels = 0
        split_objects = 0
        for image in images.values():
            file_name = str(image.get("file_name", ""))
            source_image = find_image_path(annotation_path, file_name)
            target_image = image_dir / Path(file_name).name
            if source_image is None:
                summary["missing_images"].append(str(annotation_path.parent / file_name))
            elif copy_images:
                shutil.copy2(source_image, target_image)
                split_copied += 1
            else:
                target_image = source_image

            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if (width <= 0 or height <= 0) and source_image is not None:
                img = read_image(source_image)
                height, width = img.shape[:2]

            label_lines: list[str] = []
            for annotation in annotations_by_image.get(image.get("id"), []):
                segmentation = annotation.get("segmentation")
                polygons = segmentation if isinstance(segmentation, list) else []
                object_written = False
                for polygon in polygons:
                    if not isinstance(polygon, list) or len(polygon) < 6 or width <= 0 or height <= 0:
                        continue
                    coords = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] = np.clip(coords[:, 0] / float(width), 0.0, 1.0)
                    coords[:, 1] = np.clip(coords[:, 1] / float(height), 0.0, 1.0)
                    flat = " ".join(f"{value:.6f}" for value in coords.reshape(-1))
                    label_lines.append(f"0 {flat}")
                    object_written = True
                if object_written:
                    split_objects += 1
                else:
                    summary["skipped_annotations"] += 1

            (label_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(label_lines))
            split_labels += 1

        summary["splits"][split] = {
            "source_split": source_split,
            "images": len(images),
            "copied_images": split_copied,
            "label_files": split_labels,
            "objects": split_objects,
        }

    yaml_lines = [f"path: {output.resolve().as_posix()}"]
    if "train" in present_splits:
        yaml_lines.append("train: images/train")
    if "val" in present_splits:
        yaml_lines.append("val: images/val")
    elif "train" in present_splits:
        yaml_lines.append("val: images/train")
    if "test" in present_splits:
        yaml_lines.append("test: images/test")
    yaml_lines.extend(["names:", f"  0: {class_name}"])
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    return summary


def _range_stats(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }

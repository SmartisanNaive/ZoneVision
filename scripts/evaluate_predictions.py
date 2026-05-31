from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from zonevision.coco import iter_coco_instances, segmentation_to_mask
from zonevision.io_utils import read_image
from zonevision.pipeline import resolve_workspace_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ZoneVision predicted masks against COCO polygon annotations.")
    parser.add_argument("--coco", required=True, help="COCO/Roboflow dataset root.")
    parser.add_argument("--predictions", required=True, help="Pipeline output directory containing measurements.csv and masks/.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matched masks.")
    args = parser.parse_args()

    coco_root = resolve_workspace_path(args.coco)
    predictions_dir = resolve_workspace_path(args.predictions)
    summary = evaluate_predictions(coco_root, predictions_dir, args.iou_threshold)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output_path = resolve_workspace_path(args.output, must_exist=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def evaluate_predictions(coco_root: Path, predictions_dir: Path, iou_threshold: float) -> dict[str, object]:
    measurements_path = predictions_dir / "measurements.csv"
    rows = _read_measurements(measurements_path)
    gt_by_image: dict[str, list] = {}
    for instance in iter_coco_instances(coco_root):
        gt_by_image.setdefault(Path(instance.image_file).name, []).append(instance)

    by_image: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_image.setdefault(Path(str(row.get("image_name", ""))).name, []).append(row)

    per_image: list[dict[str, object]] = []
    all_ious: list[float] = []
    total_gt = 0
    total_pred = 0
    total_matched = 0

    image_names = sorted(by_image) if by_image else sorted(gt_by_image)
    for image_name in image_names:
        gt_instances = gt_by_image.get(image_name, [])
        pred_rows = by_image.get(image_name, [])
        if not gt_instances and not pred_rows:
            continue
        gt_masks = [_gt_mask(instance) for instance in gt_instances]
        gt_masks = [mask for mask in gt_masks if mask is not None]
        pred_masks = [_pred_mask(row) for row in pred_rows]
        pred_masks = [mask for mask in pred_masks if mask is not None]
        matches, ious = greedy_match(gt_masks, pred_masks, iou_threshold)
        all_ious.extend(ious)
        total_gt += len(gt_masks)
        total_pred += len(pred_masks)
        total_matched += matches
        per_image.append(
            {
                "image_name": image_name,
                "gt_instances": len(gt_masks),
                "pred_instances": len(pred_masks),
                "matched_instances": matches,
                "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else None,
            }
        )

    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "coco": str(coco_root),
        "predictions": str(predictions_dir),
        "iou_threshold": iou_threshold,
        "totals": {
            "gt_instances": total_gt,
            "pred_instances": total_pred,
            "matched_instances": total_matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_matched_iou": round(float(np.mean(all_ious)), 4) if all_ious else None,
        },
        "images": per_image,
    }


def greedy_match(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], threshold: float) -> tuple[int, list[float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            if gt_mask.shape != pred_mask.shape:
                pred_mask = _resize_bool(pred_mask, gt_mask.shape)
            iou = mask_iou(gt_mask, pred_mask)
            if iou >= threshold:
                candidates.append((iou, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_ious: list[float] = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matched_ious.append(float(iou))
    return len(matched_ious), matched_ious


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def _gt_mask(instance) -> np.ndarray | None:
    return segmentation_to_mask(instance.segmentation, (instance.height, instance.width), source_size=(instance.width, instance.height))


def _pred_mask(row: dict[str, str]) -> np.ndarray | None:
    path = row.get("mask_path")
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        return None
    image = read_image(mask_path)
    return image[:, :, 0] > 127


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _read_measurements(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())

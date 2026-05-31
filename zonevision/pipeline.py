from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ZoneVisionConfig
from .glm_qc import run_glm_visual_qc, should_review_image, write_visual_qc_files
from .io_utils import discover_images, ensure_dir, read_image, write_csv, write_image, write_mask
from .plate import PlateGeometry, WellPoint, estimate_plate_geometry, to_gray
from .postprocess import build_well_phenotype_rows, write_well_phenotype_files
from .rfdetr_integration import create_rfdetr_model, predict_rfdetr_seg

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float
    confidence: float
    source: str
    row: int | None
    col: int | None
    mask: np.ndarray | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ZoneVision annotation pipeline.")
    parser.add_argument("--input", required=True, help="Input dataset directory.")
    parser.add_argument("--output", default="ZoneVision/results", help="Output directory.")
    parser.add_argument("--config", default="ZoneVision/config.yaml", help="Config YAML path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for quick tests.")
    parser.add_argument("--detector", choices=("yolo", "rfdetr", "auto"), default=None, help="Candidate detector to use.")
    parser.add_argument("--disable-yolo", action="store_true", help="Skip YOLO and use bootstrap only.")
    parser.add_argument("--disable-rfdetr", action="store_true", help="Skip RF-DETR and use other configured detectors.")
    parser.add_argument("--disable-sam3", action="store_true", help="Skip SAM3 refinement.")
    parser.add_argument("--yolo-model", default=None, help="Override the YOLO model or checkpoint path.")
    parser.add_argument("--rfdetr-model", default=None, help="Override the RF-DETR checkpoint path.")
    parser.add_argument("--rfdetr-variant", default=None, help="Override the RF-DETR-Seg variant.")
    parser.add_argument("--rfdetr-conf", type=float, default=None, help="Override the RF-DETR confidence threshold.")
    parser.add_argument("--sam3-checkpoint", default=None, help="Override the local SAM3 checkpoint path.")
    parser.add_argument("--enable-glm-qc", action="store_true", help="Run GLM visual review and write visual_qc outputs.")
    parser.add_argument("--glm-model", default=None, help="Override the GLM visual review model.")
    parser.add_argument("--glm-api-key-env", default=None, help="Environment variable containing the GLM API key.")
    parser.add_argument("--glm-base-url", default=None, help="OpenAI-compatible GLM API base URL.")
    parser.add_argument("--glm-qc-scope", choices=("all", "review", "summary"), default=None, help="Images to review with GLM.")
    parser.add_argument("--glm-qc-max-images", type=int, default=None, help="Maximum images to review with GLM.")
    parser.add_argument("--glm-qc-timeout-s", type=float, default=None, help="GLM request timeout in seconds.")
    parser.add_argument("--glm-qc-crop-max-side", type=int, default=None, help="Maximum side length for images sent to GLM.")
    parser.add_argument("--disable-glm-original", action="store_true", help="Do not send the original image to GLM.")
    parser.add_argument("--disable-glm-overlay", action="store_true", help="Do not send the overlay image to GLM.")
    parser.add_argument("--glm-qc-action", choices=("flag_only",), default=None, help="GLM action mode; first version only records flags.")
    parser.add_argument("--device", default=None, help="Override the runtime device: auto, mps, cpu, or cuda index.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO inference image size.")
    parser.add_argument("--yolo-conf", type=float, default=None, help="Override the YOLO confidence threshold.")
    args = parser.parse_args()

    config = ZoneVisionConfig.from_yaml(resolve_workspace_path(args.config))
    config = config.merge(
        {
            "detector": args.detector,
            "enable_yolo": False if args.disable_yolo else None,
            "enable_rfdetr": False if args.disable_rfdetr else (True if args.detector in {"rfdetr", "auto"} or args.rfdetr_model else None),
            "enable_sam3": False if args.disable_sam3 else None,
            "max_images": args.limit,
            "yolo_model": args.yolo_model,
            "rfdetr_model": args.rfdetr_model,
            "rfdetr_variant": args.rfdetr_variant,
            "rfdetr_conf": args.rfdetr_conf,
            "sam3_checkpoint": args.sam3_checkpoint,
            "glm_qc_enabled": True if args.enable_glm_qc else None,
            "glm_model": args.glm_model,
            "glm_api_key_env": args.glm_api_key_env,
            "glm_base_url": args.glm_base_url,
            "glm_qc_scope": args.glm_qc_scope,
            "glm_qc_max_images": args.glm_qc_max_images,
            "glm_qc_timeout_s": args.glm_qc_timeout_s,
            "glm_qc_crop_max_side": args.glm_qc_crop_max_side,
            "glm_qc_use_original": False if args.disable_glm_original else None,
            "glm_qc_use_overlay": False if args.disable_glm_overlay else None,
            "glm_qc_action": args.glm_qc_action,
            "device": args.device,
            "imgsz": args.imgsz,
            "yolo_conf": args.yolo_conf,
        }
    )
    run_pipeline(resolve_workspace_path(args.input), resolve_workspace_path(args.output, must_exist=False), config)
    return 0


def train_main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO26 model for inhibition-zone detection.")
    parser.add_argument("--data", required=True, help="Path to the YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--project", default="ZoneVision/results/yolo_training", help="Training output project directory.")
    parser.add_argument("--name", default="zonevision_yolo26", help="Training run name.")
    parser.add_argument("--device", default="auto", help="Training device: auto, mps, cpu, or cuda index.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required for training: {exc}") from exc

    device = resolve_device(args.device)
    data_path = resolve_workspace_path(args.data).resolve()
    project_dir = resolve_workspace_path(args.project, must_exist=False).resolve()
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(project_dir),
        name=args.name,
        device=device,
    )
    return 0


def export_bootstrap_main() -> int:
    parser = argparse.ArgumentParser(description="Convert bootstrap masks into a YOLO segmentation dataset.")
    parser.add_argument("--measurements", required=True, help="Path to the measurements CSV.")
    parser.add_argument("--output", default="ZoneVision/data/yolo", help="YOLO dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for the train/val split.")
    args = parser.parse_args()

    export_bootstrap_dataset(Path(args.measurements), Path(args.output), args.val_ratio, args.seed)
    return 0


def run_pipeline(input_dir: Path, output_dir: Path, config: ZoneVisionConfig) -> None:
    images = discover_images(input_dir)
    if config.max_images is not None:
        images = images[: config.max_images]
    if not images:
        raise SystemExit(f"No images found under {input_dir}")

    overlay_dir = ensure_dir(output_dir / "overlays")
    mask_root = ensure_dir(output_dir / "masks")
    debug_dir = ensure_dir(output_dir / "debug") if config.save_debug else None
    device = resolve_device(config.device)
    yolo_model = load_yolo_model(config) if config.enable_yolo and config.detector in {"yolo", "auto"} else None
    rfdetr_model = load_rfdetr_model(config) if config.enable_rfdetr and config.detector in {"rfdetr", "auto"} else None
    if config.detector == "rfdetr" and config.enable_rfdetr and rfdetr_model is None:
        raise SystemExit("RF-DETR detector requested, but no valid RF-DETR checkpoint was loaded.")
    sam_model = load_sam_model(config, device) if config.enable_sam3 else None

    rows: list[dict[str, object]] = []
    phenotype_rows: list[dict[str, object]] = []
    visual_qc_rows: list[dict[str, object]] = []
    glm_reviews = 0
    for image_path in images:
        image = read_image(image_path)
        geometry = estimate_plate_geometry(image, config.well_pitch_mm)
        detections = detect_candidates(image, geometry, yolo_model, rfdetr_model, config, device)

        refined: list[Detection] = []
        for detection in detections:
            mask = detection.mask
            if sam_model is not None:
                refined_mask = refine_with_sam3(image, detection, sam_model, config, device)
                if refined_mask is not None and refined_mask.sum() > 0:
                    mask = refined_mask
            detection.mask = mask
            refined.append(detection)

        overlay, image_rows = build_outputs(
            image=image,
            image_path=image_path,
            detections=refined,
            geometry=geometry,
            overlay_dir=overlay_dir,
            mask_root=mask_root,
            draw_measurements=config.draw_measurements,
        )
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        write_image(overlay_path, overlay)
        image_phenotype_rows = build_well_phenotype_rows(image_path, refined, geometry, image_rows, config, measure_mask)
        rows.extend(image_rows)
        phenotype_rows.extend(image_phenotype_rows)

        if config.glm_qc_enabled and glm_reviews < config.glm_qc_max_images and should_review_image(image_phenotype_rows, config.glm_qc_scope):
            visual_qc_rows.append(
                run_glm_visual_qc(
                    image_path=image_path,
                    overlay_path=overlay_path,
                    measurement_rows=image_rows,
                    phenotype_rows=image_phenotype_rows,
                    config=config,
                )
            )
            glm_reviews += 1

        if debug_dir is not None:
            debug_view = draw_geometry(image.copy(), geometry)
            write_image(debug_dir / f"{image_path.stem}_grid.png", debug_view)

    write_csv(output_dir / "measurements.csv", rows)
    write_well_phenotype_files(output_dir, phenotype_rows)
    write_visual_qc_files(output_dir, visual_qc_rows)


def detect_candidates(
    image: np.ndarray,
    geometry: PlateGeometry,
    yolo_model,
    rfdetr_model,
    config: ZoneVisionConfig,
    device: str,
) -> list[Detection]:
    if config.detector == "rfdetr":
        if rfdetr_model is not None:
            detections = run_rfdetr_detection(image, geometry, rfdetr_model, config)
            if detections:
                return detections
    elif config.detector == "auto":
        if rfdetr_model is not None:
            detections = run_rfdetr_detection(image, geometry, rfdetr_model, config)
            if detections:
                return detections
        if yolo_model is not None:
            detections = run_yolo_detection(image, geometry, yolo_model, config, device)
            if detections:
                return detections
    else:
        if yolo_model is not None:
            detections = run_yolo_detection(image, geometry, yolo_model, config, device)
            if detections:
                return detections
    if config.fallback_to_bootstrap:
        return bootstrap_detections(image, geometry, config)
    return []


def run_yolo_detection(
    image: np.ndarray,
    geometry: PlateGeometry,
    yolo_model,
    config: ZoneVisionConfig,
    device: str,
) -> list[Detection]:
    try:
        results = yolo_model.predict(
            source=image,
            conf=config.yolo_conf,
            imgsz=config.imgsz,
            device=device,
            verbose=False,
        )
    except Exception:
        return []

    detections: list[Detection] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
    mask_data = masks.data.cpu().numpy() if masks is not None and getattr(masks, "data", None) is not None else None

    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        center_x = float((x1 + x2) / 2.0)
        center_y = float((y1 + y2) / 2.0)
        nearest = nearest_well(center_x, center_y, geometry.wells)
        if nearest is None:
            continue
        if geometry.pitch_px is not None:
            distance = float(np.hypot(center_x - nearest.x, center_y - nearest.y))
            box_size = max(x2 - x1, y2 - y1)
            if distance > geometry.pitch_px * 0.45:
                continue
            if box_size > geometry.pitch_px * 1.25:
                continue
        instance_mask = None
        if config.use_yolo_seg and mask_data is not None and index < len(mask_data):
            instance_mask = mask_data[index] > 0.5
        detections.append(
            Detection(
                bbox=clip_bbox((x1, y1, x2, y2), image.shape[:2]),
                center_x=center_x,
                center_y=center_y,
                confidence=float(confs[index]),
                source="yolo",
                row=nearest.row,
                col=nearest.col,
                mask=instance_mask,
            )
        )
    return detections


def run_rfdetr_detection(
    image: np.ndarray,
    geometry: PlateGeometry,
    rfdetr_model,
    config: ZoneVisionConfig,
) -> list[Detection]:
    try:
        results = predict_rfdetr_seg(image, rfdetr_model, config.rfdetr_conf)
    except Exception as exc:
        if config.detector == "rfdetr":
            raise RuntimeError(f"RF-DETR inference failed: {exc}") from exc
        return []

    xyxy = getattr(results, "xyxy", None)
    if xyxy is None:
        return []

    confidences = getattr(results, "confidence", None)
    masks = getattr(results, "mask", None)
    detections: list[Detection] = []
    height, width = image.shape[:2]

    for index, box in enumerate(np.asarray(xyxy)):
        if index >= config.rfdetr_max_detections:
            break
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        bbox = clip_bbox((x1, y1, x2, y2), image.shape[:2])
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float((bbox[1] + bbox[3]) / 2.0)
        nearest = nearest_well(center_x, center_y, geometry.wells)
        if nearest is None:
            continue
        if geometry.pitch_px is not None:
            distance = float(np.hypot(center_x - nearest.x, center_y - nearest.y))
            box_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if distance > geometry.pitch_px * 0.45:
                continue
            if box_size > geometry.pitch_px * 1.25:
                continue

        instance_mask = None
        if masks is not None and index < len(masks):
            instance_mask = np.asarray(masks[index]) > 0
            if instance_mask.shape[:2] != (height, width):
                instance_mask = cv2.resize(
                    instance_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

        confidence = 1.0
        if confidences is not None and index < len(confidences):
            confidence = float(confidences[index])
        detections.append(
            Detection(
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                confidence=confidence,
                source="rfdetr",
                row=nearest.row,
                col=nearest.col,
                mask=instance_mask,
            )
        )

    return dedupe_detections_by_well(detections, geometry)



def dedupe_detections_by_well(detections: list[Detection], geometry: PlateGeometry) -> list[Detection]:
    best: dict[tuple[int | None, int | None], tuple[float, Detection]] = {}
    for detection in detections:
        key = (detection.row, detection.col)
        nearest = nearest_well(detection.center_x, detection.center_y, geometry.wells)
        distance = float(np.hypot(detection.center_x - nearest.x, detection.center_y - nearest.y)) if nearest else 0.0
        score = detection.confidence - distance * 1e-4
        if key not in best or score > best[key][0]:
            best[key] = (score, detection)
    return [item[1] for item in sorted(best.values(), key=lambda item: (item[1].row is None, item[1].row or 0, item[1].col or 0))]



def bootstrap_detections(image: np.ndarray, geometry: PlateGeometry, config: ZoneVisionConfig) -> list[Detection]:
    if not geometry.wells:
        return []

    gray = to_gray(image).astype(np.float32)
    fallback_pitch = geometry.pitch_px or (min(image.shape[:2]) / 12.0)
    fallback_radius = geometry.well_radius_px or (fallback_pitch * 0.18)
    px_per_mm = geometry.px_per_mm or (fallback_pitch / max(config.well_pitch_mm, 1e-6))
    contrast_map = zone_contrast_map(gray, fallback_radius)
    proposals: list[tuple[float, Detection]] = []

    for well in geometry.wells:
        search_radius = min(fallback_pitch * 0.48, config.max_zone_radius_mm * px_per_mm)
        inner_radius = max(fallback_radius * 1.2, config.min_zone_radius_mm * px_per_mm)
        zone_radius, peak_delta = estimate_zone_radius(contrast_map, well.x, well.y, inner_radius, search_radius)
        if zone_radius is None:
            continue

        x1 = int(round(well.x - zone_radius))
        y1 = int(round(well.y - zone_radius))
        x2 = int(round(well.x + zone_radius))
        y2 = int(round(well.y + zone_radius))
        bbox = clip_bbox((x1, y1, x2, y2), image.shape[:2])
        mask = circular_mask(image.shape[:2], well.x, well.y, zone_radius)
        proposals.append(
            (
                float(peak_delta),
                Detection(
                    bbox=bbox,
                    center_x=well.x,
                    center_y=well.y,
                    confidence=0.0,
                    source="bootstrap",
                    row=well.row,
                    col=well.col,
                    mask=mask,
                ),
            )
        )
    if not proposals:
        return []

    peaks = np.asarray([peak for peak, _ in proposals], dtype=np.float32)
    q25 = float(np.quantile(peaks, 0.25))
    q75 = float(np.quantile(peaks, 0.75))
    adaptive_threshold = max(config.bootstrap_intensity_delta, q75 + 0.5 * (q75 - q25))
    filtered = [(peak, detection) for peak, detection in proposals if peak >= adaptive_threshold]
    if len(filtered) > int(len(geometry.wells) * 0.35):
        limit = max(8, int(round(len(geometry.wells) * 0.2)))
        filtered = sorted(filtered, key=lambda item: item[0], reverse=True)[:limit]

    detections: list[Detection] = []
    for peak_delta, detection in filtered:
        confidence = min(0.95, 0.45 + (peak_delta / 40.0))
        detections.append(
            Detection(
                bbox=detection.bbox,
                center_x=detection.center_x,
                center_y=detection.center_y,
                confidence=float(confidence),
                source="bootstrap",
                row=detection.row,
                col=detection.col,
                mask=detection.mask,
            )
        )
    return detections


def zone_contrast_map(gray: np.ndarray, fallback_radius: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=max(4.0, fallback_radius * 0.8)) - gray


def estimate_zone_radius(
    gray: np.ndarray,
    center_x: float,
    center_y: float,
    inner_radius: float,
    search_radius: float,
) -> tuple[float | None, float]:
    pad = int(np.ceil(search_radius)) + 8
    x1 = max(0, int(center_x) - pad)
    y1 = max(0, int(center_y) - pad)
    x2 = min(gray.shape[1], int(center_x) + pad + 1)
    y2 = min(gray.shape[0], int(center_y) + pad + 1)
    roi = gray[y1:y2, x1:x2]
    yy, xx = np.indices(roi.shape)
    distances = np.hypot(xx + x1 - center_x, yy + y1 - center_y)
    ring_ids = distances.astype(np.int32)
    max_ring = int(min(search_radius, ring_ids.max()))
    if max_ring <= int(inner_radius) + 3:
        return None, 0.0

    mask = ring_ids <= max_ring
    counts = np.bincount(ring_ids[mask].ravel())
    sums = np.bincount(ring_ids[mask].ravel(), weights=roi[mask].ravel())
    valid = counts > 0
    profile = np.zeros_like(sums, dtype=np.float32)
    profile[valid] = sums[valid] / counts[valid]
    profile = smooth_1d(profile, window=5)

    start = int(max(1, inner_radius))
    stop = max_ring
    usable = profile[start : stop + 1]
    peak = float(usable.max()) if usable.size else 0.0
    if peak <= 0:
        return None, 0.0
    peak_local_index = int(np.argmax(usable))
    tail = usable[peak_local_index:]
    threshold = max(peak * 0.45, 1.5)
    below = np.where(tail < threshold)[0]
    if below.size == 0:
        return None, peak
    radius = float(start + peak_local_index + int(below[0]))
    if radius <= inner_radius + 2:
        return None, peak
    return radius, peak


def refine_with_sam3(
    image: np.ndarray,
    detection: Detection,
    sam_model,
    config: ZoneVisionConfig,
    device: str,
) -> np.ndarray | None:
    x1, y1, x2, y2 = expand_bbox(detection.bbox, image.shape[:2], config.roi_expand_ratio)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return detection.mask

    scale = 1.0
    max_side = max(roi.shape[:2])
    if max_side > config.sam3_roi_max_side:
        scale = config.sam3_roi_max_side / float(max_side)
        new_w = max(1, int(round(roi.shape[1] * scale)))
        new_h = max(1, int(round(roi.shape[0] * scale)))
        roi_for_model = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        roi_for_model = roi

    local_box = np.array(
        [
            max(1, int(round((detection.bbox[0] - x1) * scale))),
            max(1, int(round((detection.bbox[1] - y1) * scale))),
            min(roi_for_model.shape[1] - 2, int(round((detection.bbox[2] - x1) * scale))),
            min(roi_for_model.shape[0] - 2, int(round((detection.bbox[3] - y1) * scale))),
        ]
    )
    sam_imgsz = round_up_to_multiple(max(336, max(roi_for_model.shape[:2])), 14)

    try:
        results = sam_model.predict(
            source=roi_for_model,
            bboxes=local_box.tolist(),
            imgsz=sam_imgsz,
            device=device,
            verbose=False,
        )
    except Exception:
        return detection.mask

    if not results:
        return detection.mask
    masks = getattr(results[0], "masks", None)
    if masks is None or getattr(masks, "data", None) is None or len(masks.data) == 0:
        return detection.mask

    roi_mask = masks.data[0].cpu().numpy() > 0.5
    if scale != 1.0:
        roi_mask = cv2.resize(roi_mask.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    full_mask = np.zeros(image.shape[:2], dtype=bool)
    full_mask[y1:y2, x1:x2] = roi_mask
    if detection.mask is not None and detection.mask.sum() > 0:
        if config.sam_mask_merge_policy == "union":
            full_mask = np.logical_or(full_mask, detection.mask)
        elif config.sam_mask_merge_policy == "intersect_if_overlap" and np.logical_and(full_mask, detection.mask).any():
            full_mask = np.logical_and(full_mask, detection.mask)
    return full_mask


def build_outputs(
    image: np.ndarray,
    image_path: Path,
    detections: list[Detection],
    geometry: PlateGeometry,
    overlay_dir: Path,
    mask_root: Path,
    draw_measurements: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    overlay = image.copy()
    rows: list[dict[str, object]] = []
    px_per_mm = geometry.px_per_mm
    image_mask_dir = ensure_dir(mask_root / image_path.stem)

    for index, detection in enumerate(detections, start=1):
        if detection.mask is None or detection.mask.sum() == 0:
            continue
        color = indexed_color(index)
        overlay = alpha_fill(overlay, detection.mask, color, 0.35)
        contour = mask_contour(detection.mask)
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, color, 2)

        metrics = measure_mask(detection.mask, px_per_mm)
        label = f"{index}"
        if draw_measurements and metrics["diameter_eq_mm"] is not None:
            label = f"{index}:{metrics['diameter_eq_mm']:.1f}mm"
        text_origin = (int(round(detection.center_x)), int(round(detection.center_y)))
        cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        mask_path = image_mask_dir / f"zone_{index:03d}.png"
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        write_mask(mask_path, detection.mask)
        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "instance_id": index,
                "row": detection.row,
                "col": detection.col,
                "confidence": round(detection.confidence, 4),
                "source": detection.source,
                "center_x_px": round(detection.center_x, 2),
                "center_y_px": round(detection.center_y, 2),
                "area_px": int(metrics["area_px"]),
                "diameter_eq_px": round(metrics["diameter_eq_px"], 2),
                "diameter_max_px": round(metrics["diameter_max_px"], 2),
                "px_per_mm": round(px_per_mm, 4) if px_per_mm is not None else None,
                "diameter_eq_mm": round(metrics["diameter_eq_mm"], 3) if metrics["diameter_eq_mm"] is not None else None,
                "diameter_max_mm": round(metrics["diameter_max_mm"], 3) if metrics["diameter_max_mm"] is not None else None,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
    return overlay, rows


def measure_mask(mask: np.ndarray, px_per_mm: float | None) -> dict[str, float | int | None]:
    area_px = int(mask.sum())
    diameter_eq_px = float(2.0 * np.sqrt(area_px / np.pi)) if area_px > 0 else 0.0
    contour = mask_contour(mask)
    diameter_max_px = contour_max_distance(contour) if contour is not None else 0.0
    diameter_eq_mm = (diameter_eq_px / px_per_mm) if px_per_mm else None
    diameter_max_mm = (diameter_max_px / px_per_mm) if px_per_mm else None
    return {
        "area_px": area_px,
        "diameter_eq_px": diameter_eq_px,
        "diameter_max_px": diameter_max_px,
        "diameter_eq_mm": diameter_eq_mm,
        "diameter_max_mm": diameter_max_mm,
    }


def export_bootstrap_dataset(measurements_csv: Path, output_dir: Path, val_ratio: float, seed: int) -> None:
    with measurements_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {measurements_csv}")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["image_path"], []).append(row)

    output_dir = ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    labels_dir = ensure_dir(output_dir / "labels")
    for split in ("train", "val"):
        ensure_dir(images_dir / split)
        ensure_dir(labels_dir / split)

    image_paths = sorted(Path(path) for path in grouped)
    random.Random(seed).shuffle(image_paths)
    val_count = max(1, int(round(len(image_paths) * val_ratio))) if len(image_paths) > 1 else 0
    val_set = {path for path in image_paths[:val_count]}

    for image_path in image_paths:
        split = "val" if image_path in val_set else "train"
        target_image = images_dir / split / image_path.name
        shutil.copy2(image_path, target_image)
        image = read_image(image_path)
        height, width = image.shape[:2]

        label_lines: list[str] = []
        for row in grouped[str(image_path)]:
            mask = read_image(row["mask_path"])
            gray = to_gray(mask)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = max(1.5, 0.0025 * cv2.arcLength(contour, True))
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            normalized = []
            for x, y in polygon:
                normalized.append(f"{x / width:.6f}")
                normalized.append(f"{y / height:.6f}")
            label_lines.append("0 " + " ".join(normalized))

        target_label = labels_dir / split / f"{image_path.stem}.txt"
        target_label.write_text("\n".join(label_lines))

    yaml_path = output_dir.parent / "yolo_dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: inhibition_zone",
            ]
        )
        + "\n"
    )


def load_yolo_model(config: ZoneVisionConfig):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    try:
        return YOLO(config.yolo_model)
    except Exception:
        return None


def load_rfdetr_model(config: ZoneVisionConfig):
    if not config.rfdetr_model:
        return None
    checkpoint = resolve_workspace_path(config.rfdetr_model)
    if not checkpoint.exists():
        return None
    try:
        return create_rfdetr_model(config.rfdetr_variant, checkpoint)
    except Exception:
        return None



def load_sam_model(config: ZoneVisionConfig, device: str):
    checkpoint = resolve_workspace_path(config.sam3_checkpoint)
    if not checkpoint.exists():
        return None
    try:
        from ultralytics import SAM
    except Exception:
        return None
    try:
        return SAM(str(checkpoint))
    except Exception:
        return None


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def resolve_workspace_path(path: str | Path, must_exist: bool = True) -> Path:
    candidate = Path(path)
    candidates = [candidate]

    if candidate.parts and candidate.parts[0] == PROJECT_ROOT.name:
        candidates.append(PROJECT_ROOT / Path(*candidate.parts[1:]))

    candidates.append(PROJECT_ROOT / candidate)
    candidates.append(PROJECT_ROOT.parent / candidate)

    seen: set[Path] = set()
    for option in candidates:
        if option in seen:
            continue
        seen.add(option)
        if not must_exist or option.exists():
            return option
    return candidate


def nearest_well(center_x: float, center_y: float, wells: list[WellPoint]) -> WellPoint | None:
    if not wells:
        return None
    distances = [np.hypot(center_x - well.x, center_y - well.y) for well in wells]
    return wells[int(np.argmin(distances))]


def clip_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = int(round(width * expand_ratio))
    pad_y = int(round(height * expand_ratio))
    return clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), shape)


def circular_mask(shape: tuple[int, int], center_x: float, center_y: float, radius: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    return np.hypot(xx - center_x, yy - center_y) <= radius


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def round_up_to_multiple(value: int, divisor: int) -> int:
    return ((int(value) + divisor - 1) // divisor) * divisor


def alpha_fill(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    output[mask] = (output[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return output


def indexed_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 132),
        (54, 162, 235),
        (255, 206, 86),
        (75, 192, 192),
        (153, 102, 255),
        (255, 159, 64),
        (0, 200, 83),
        (255, 64, 129),
    ]
    return palette[(index - 1) % len(palette)]


def mask_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_max_distance(contour: np.ndarray | None) -> float:
    if contour is None or len(contour) < 2:
        return 0.0
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) > 128:
        step = max(1, len(points) // 128)
        points = points[::step]
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    return float(distances.max())


def draw_geometry(image: np.ndarray, geometry: PlateGeometry) -> np.ndarray:
    for well in geometry.wells:
        cv2.circle(image, (int(round(well.x)), int(round(well.y))), int(round(well.radius)), (0, 255, 255), 1)
    return image

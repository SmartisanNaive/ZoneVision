from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SEGMENTATION_VARIANTS = {
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
    "seg-2xlarge": "RFDETRSeg2XLarge",
}


def create_rfdetr_model(variant: str = "seg-small", checkpoint: str | Path | None = None):
    try:
        import rfdetr
    except Exception as exc:
        raise RuntimeError(f"RF-DETR is required: {exc}") from exc

    class_name = SEGMENTATION_VARIANTS.get(variant)
    if class_name is None:
        choices = ", ".join(sorted(SEGMENTATION_VARIANTS))
        raise ValueError(f"Unsupported RF-DETR variant '{variant}'. Choose one of: {choices}")

    model_class = getattr(rfdetr, class_name)
    if checkpoint:
        return model_class(pretrain_weights=str(checkpoint))
    return model_class()


def train_rfdetr_seg(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str = "seg-small",
    epochs: int = 100,
    batch_size: int = 1,
    grad_accum_steps: int = 16,
    lr: float = 1e-4,
    device: str = "auto",
    seed: int = 13,
) -> None:
    model = create_rfdetr_model(variant)
    kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "output_dir": str(output_dir),
        "seed": seed,
    }
    if device and device != "auto":
        kwargs["device"] = device
    model.train(**kwargs)


def predict_rfdetr_seg(image_bgr: np.ndarray, model, threshold: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return model.predict(image_rgb, threshold=threshold)


def train_rfdetr_main() -> int:
    from .pipeline import resolve_device, resolve_workspace_path

    parser = argparse.ArgumentParser(description="Train an RF-DETR-Seg model for inhibition-zone segmentation.")
    parser.add_argument("--dataset", required=True, help="COCO/Roboflow dataset root.")
    parser.add_argument("--variant", default="seg-small", choices=sorted(SEGMENTATION_VARIANTS), help="RF-DETR-Seg variant.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-step batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--output", default="ZoneVision/results/rfdetr_training/yijunquan_seg_small", help="Training output directory.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, mps, or CUDA device.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    args = parser.parse_args()

    train_rfdetr_seg(
        resolve_workspace_path(args.dataset),
        resolve_workspace_path(args.output, must_exist=False),
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        device=resolve_device(args.device),
        seed=args.seed,
    )
    return 0

from __future__ import annotations

import argparse
import json

from zonevision.coco import normalize_coco_dataset
from zonevision.pipeline import resolve_workspace_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Roboflow COCO annotations for RF-DETR training.")
    parser.add_argument("--input", required=True, help="Input COCO/Roboflow dataset root.")
    parser.add_argument("--output", required=True, help="Output normalized dataset root.")
    parser.add_argument("--class-name", default="inhibition_zone", help="Single class name to write into COCO categories.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional image-level train split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=None, help="Optional image-level validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Optional image-level test split ratio.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for optional image-level splitting.")
    args = parser.parse_args()

    summary = normalize_coco_dataset(
        resolve_workspace_path(args.input),
        resolve_workspace_path(args.output, must_exist=False),
        class_name=args.class_name,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json

from zonevision.coco import export_yolo_seg_dataset
from zonevision.pipeline import resolve_workspace_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export COCO polygon annotations to Ultralytics YOLO segmentation format.")
    parser.add_argument("--dataset", required=True, help="Input COCO/Roboflow dataset root.")
    parser.add_argument("--output", required=True, help="Output YOLO segmentation dataset directory.")
    parser.add_argument("--class-name", default="inhibition_zone", help="Class name to write in data.yaml.")
    parser.add_argument("--exclude-test", action="store_true", help="Skip test split when exporting labels.")
    args = parser.parse_args()

    summary = export_yolo_seg_dataset(
        resolve_workspace_path(args.dataset),
        resolve_workspace_path(args.output, must_exist=False),
        class_name=args.class_name,
        include_test=not args.exclude_test,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

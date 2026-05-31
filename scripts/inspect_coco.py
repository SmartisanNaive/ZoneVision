from __future__ import annotations

import argparse
import json

from zonevision.coco import inspect_coco_dataset
from zonevision.pipeline import resolve_workspace_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a COCO/Roboflow annotation dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset root or COCO JSON path.")
    args = parser.parse_args()

    summary = inspect_coco_dataset(resolve_workspace_path(args.dataset))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

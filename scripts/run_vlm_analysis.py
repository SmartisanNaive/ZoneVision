"""VLM-based consistency analysis and failure case analysis for paper benchmark.

Uses GLM-5V to:
1. Assess per-image agreement between algorithm segmentation and visual ground truth (consistency review)
2. Analyze FP/FN failure cases with qualitative explanations
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import cv2
import httpx
import numpy as np

from zonevision.coco import segmentation_to_mask
from zonevision.io_utils import read_image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_dotenv() -> str:
    for p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("GLM_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


API_KEY = os.environ.get("GLM_API_KEY", "") or _load_dotenv()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5v-turbo"
TIMEOUT = 60
MAX_SIDE = 1024


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------
def encode_image(path: Path, max_side: int = MAX_SIDE) -> str:
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return base64.b64encode(data).decode()
    h, w = img.shape[:2]
    if max(h, w) > max_side > 0:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(enc.tobytes() if ok else data).decode()


def call_glm(prompt: str, images: list[Path], max_retries: int = 2) -> dict:
    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        if img_path.exists():
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"},
            })
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(BASE_URL, headers=headers, json=payload)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return _parse_json(text)
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(3)
            else:
                return {"error": str(exc), "raw": text if 'text' in dir() else ""}


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
        return {"raw_text": text[:2000]}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def roboflow_to_original(fn: str) -> str:
    m = re.match(r"(.+)_jpg\.rf\.[a-zA-Z0-9]+\.jpg", fn)
    return m.group(1).replace("_jpg", "") + ".jpg" if m else fn


def load_gt(gt_json: Path):
    data = json.loads(gt_json.read_text())
    images = {img["id"]: img for img in data["images"]}
    by_fn = defaultdict(list)
    for ann in data["annotations"]:
        by_fn[images[ann["image_id"]]["file_name"]].append(ann)
    return by_fn


def load_predictions(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    by_image = defaultdict(list)
    for row in rows:
        by_image[Path(row["image_name"]).name].append(row)
    return by_image


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def consistency_prompt(img_name: str, gt_count: int, pred_count: int) -> str:
    return (
        f"你是一名抑菌圈图像分析专家。请对以下平板照片（第1张）和算法自动分割叠加图（第2张）做一致性评估。\n"
        f"图片文件名：{img_name}\n"
        f"人工标注抑菌圈数量：{gt_count}\n"
        f"算法检测抑菌圈数量：{pred_count}\n\n"
        f"请评估：\n"
        f"1. 算法分割的抑菌圈位置是否与原图中实际可见的抑菌圈一致？\n"
        f"2. 分割边界是否贴合抑菌圈实际边缘？\n"
        f"3. 是否存在明显漏检（人工标了但算法没检测到）？\n"
        f"4. 是否存在明显过分割（算法检测了但原图中不明显）？\n"
        f"5. 整体一致性评分（1-5分，5=完全一致）\n\n"
        f"严格返回JSON，不要Markdown代码块：\n"
        f'{{"image_name":"{img_name}","boundary_quality":"good|moderate|poor",'
        f'"position_agreement":"good|moderate|poor",'
        f'"missed_zones_estimate":数字,"over_segmented_estimate":数字,'
        f'"consistency_score":数字1到5,"overall_verdict":"consistent|minor_issues|major_issues",'
        f'"detail":一两句中文描述}}'
    )


def failure_prompt(img_name: str, fp_count: int, fn_count: int, gt_count: int, pred_count: int) -> str:
    return (
        f"你是一名抑菌圈图像分析专家。以下平板照片中，算法与人工标注存在差异，请分析失败原因。\n"
        f"图片文件名：{img_name}\n"
        f"人工标注：{gt_count}个，算法检测：{pred_count}个\n"
        f"漏检（FN）：约{fn_count}个，误检（FP）：约{fp_count}个\n\n"
        f"请仔细观察原图和叠加图，分析：\n"
        f"1. 漏检的抑菌圈可能在哪里？特征是什么？（如：面积小、边界模糊、与邻近圈粘连、位于边缘）\n"
        f"2. 误检的可能原因？（如：培养基纹理、气泡、光照不均、非抑菌圈圆形结构）\n"
        f"3. 这些失败是否可接受？属于系统性问题还是边缘情况？\n\n"
        f"严格返回JSON，不要Markdown代码块：\n"
        f'{{"image_name":"{img_name}","fn_analysis":{{"estimated_count":数字,'
        f'"likely_causes":["原因1","原因2"],"fn_severity":"minor|moderate|severe"}},'
        f'"fp_analysis":{{"estimated_count":数字,"likely_causes":["原因1","原因2"],'
        f'"fp_severity":"minor|moderate|severe"}},'
        f'"systemic_issue":true或false,'
        f'"recommendation":"一两句改进建议",'
        f'"detail":"一段详细中文分析"}}'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(
    gt_json: str,
    pred_csv: str,
    overlay_dir: str,
    original_dir: str,
    output_dir: str,
):
    gt_path = Path(gt_json)
    pred_path = Path(pred_csv)
    overlay_path = Path(overlay_dir)
    original_path = Path(original_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not API_KEY:
        raise RuntimeError("GLM_API_KEY not set")

    gt_by_fn = load_gt(gt_path)
    pred_by_image = load_predictions(pred_path)

    # Build mapping
    mapping = {}
    for fn in gt_by_fn:
        orig = roboflow_to_original(fn)
        mapping[orig] = fn

    # Find overlay files
    def find_overlay(img_name):
        stem = Path(img_name).stem
        for f in overlay_path.glob("*.png"):
            if stem in f.stem:
                return f
        for f in overlay_path.glob("*.jpg"):
            if stem in f.stem:
                return f
        return None

    def find_original(img_name):
        p = original_path / img_name
        if p.exists():
            return p
        for f in original_path.glob("*.jpg"):
            if Path(img_name).stem in f.stem:
                return f
        return None

    consistency_results = []
    failure_results = []

    annotated_images = sorted(mapping.keys())

    for i, img_name in enumerate(annotated_images):
        gt_fn = mapping[img_name]
        gt_anns = gt_by_fn[gt_fn]
        gt_count = len(gt_anns)
        preds = pred_by_image.get(img_name, [])
        pred_count = len(preds)

        orig_img = find_original(img_name)
        overlay_img = find_overlay(img_name)

        if not orig_img or not overlay_img:
            print(f"[{i + 1}/{len(annotated_images)}] SKIP {img_name} (images not found)")
            continue

        print(f"[{i + 1}/{len(annotated_images)}] {img_name}: GT={gt_count} Pred={pred_count}")

        # --- Consistency analysis (every image) ---
        print(f"  Running consistency analysis...")
        c_prompt = consistency_prompt(img_name, gt_count, pred_count)
        c_result = call_glm(c_prompt, [orig_img, overlay_img])
        c_result["_image"] = img_name
        c_result["_gt_count"] = gt_count
        c_result["_pred_count"] = pred_count
        consistency_results.append(c_result)
        time.sleep(1)

        # --- Failure analysis (only images with mismatches) ---
        diff = pred_count - gt_count
        if diff != 0:
            fn_count = max(0, -diff)
            fp_count = max(0, diff)
            # More accurate: use actual TP/FP/FN from benchmark
            # For now use the diff
            print(f"  Running failure analysis (FP≈{fp_count}, FN≈{fn_count})...")
            f_prompt = failure_prompt(img_name, fp_count, fn_count, gt_count, pred_count)
            f_result = call_glm(f_prompt, [orig_img, overlay_img])
            f_result["_image"] = img_name
            f_result["_gt_count"] = gt_count
            f_result["_pred_count"] = pred_count
            f_result["_fp"] = fp_count
            f_result["_fn"] = fn_count
            failure_results.append(f_result)
            time.sleep(1)

    # --- Save results ---
    report = {
        "consistency_analysis": consistency_results,
        "failure_analysis": failure_results,
        "summary": {
            "total_images_reviewed": len(consistency_results),
            "images_with_failures": len(failure_results),
            "images_perfect_match": len(consistency_results) - len(failure_results),
        },
    }

    (out / "vlm_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- CSV exports ---
    # Consistency CSV
    c_rows = []
    for r in consistency_results:
        c_rows.append({
            "image": r.get("_image", ""),
            "gt_count": r.get("_gt_count", ""),
            "pred_count": r.get("_pred_count", ""),
            "boundary_quality": r.get("boundary_quality", ""),
            "position_agreement": r.get("position_agreement", ""),
            "missed_zones_estimate": r.get("missed_zones_estimate", ""),
            "over_segmented_estimate": r.get("over_segmented_estimate", ""),
            "consistency_score": r.get("consistency_score", ""),
            "overall_verdict": r.get("overall_verdict", ""),
            "detail": r.get("detail", ""),
        })
    if c_rows:
        with open(out / "vlm_consistency.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(c_rows[0].keys()))
            w.writeheader()
            w.writerows(c_rows)

    # Failure CSV
    f_rows = []
    for r in failure_results:
        fn_a = r.get("fn_analysis", {})
        fp_a = r.get("fp_analysis", {})
        f_rows.append({
            "image": r.get("_image", ""),
            "gt_count": r.get("_gt_count", ""),
            "pred_count": r.get("_pred_count", ""),
            "fp": r.get("_fp", ""),
            "fn": r.get("_fn", ""),
            "fn_causes": "; ".join(fn_a.get("likely_causes", [])) if isinstance(fn_a, dict) else "",
            "fn_severity": fn_a.get("fn_severity", "") if isinstance(fn_a, dict) else "",
            "fp_causes": "; ".join(fp_a.get("likely_causes", [])) if isinstance(fp_a, dict) else "",
            "fp_severity": fp_a.get("fp_severity", "") if isinstance(fp_a, dict) else "",
            "systemic_issue": r.get("systemic_issue", ""),
            "recommendation": r.get("recommendation", ""),
            "detail": r.get("detail", ""),
        })
    if f_rows:
        with open(out / "vlm_failure_analysis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(f_rows[0].keys()))
            w.writeheader()
            w.writerows(f_rows)

    # --- Summary for paper ---
    scores = [r.get("consistency_score", 0) for r in consistency_results if isinstance(r.get("consistency_score"), (int, float))]
    verdicts = [r.get("overall_verdict", "") for r in consistency_results]
    paper_summary = {
        "n_images": len(consistency_results),
        "mean_consistency_score": round(float(np.mean(scores)), 2) if scores else None,
        "verdict_counts": {v: verdicts.count(v) for v in set(verdicts)},
        "n_images_with_failures": len(failure_results),
        "failure_details": [
            {
                "image": r.get("_image", ""),
                "fn_causes": r.get("fn_analysis", {}).get("likely_causes", []) if isinstance(r.get("fn_analysis"), dict) else [],
                "fp_causes": r.get("fp_analysis", {}).get("likely_causes", []) if isinstance(r.get("fp_analysis"), dict) else [],
                "systemic": r.get("systemic_issue", ""),
            }
            for r in failure_results
        ],
    }
    (out / "vlm_paper_summary.json").write_text(json.dumps(paper_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== VLM Analysis Complete ===")
    print(f"Images reviewed: {len(consistency_results)}")
    print(f"Mean consistency score: {paper_summary['mean_consistency_score']}")
    print(f"Verdicts: {paper_summary['verdict_counts']}")
    print(f"Failure cases analyzed: {len(failure_results)}")
    print(f"Output: {out}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="../Data_Base/Zone of inhibition.coco/train/_annotations.coco.json")
    parser.add_argument("--pred-csv", default="results/benchmark_rfdetr/measurements.csv")
    parser.add_argument("--overlay-dir", default="results/benchmark_rfdetr/overlays")
    parser.add_argument("--original-dir", default="../Data_Base/Antibiotic Susceptibility Testing Zone of Inhibition Image Dataset/01_color_plate_photos")
    parser.add_argument("--output", default="benchmark_report/vlm_analysis")
    args = parser.parse_args()
    run_analysis(args.gt, args.pred_csv, args.overlay_dir, args.original_dir, args.output)

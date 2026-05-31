from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from .config import ZoneVisionConfig
from .io_utils import write_csv


VisualQcResult = dict[str, object]


def run_glm_visual_qc(
    *,
    image_path: Path,
    overlay_path: Path,
    measurement_rows: list[dict[str, object]],
    phenotype_rows: list[dict[str, object]],
    config: ZoneVisionConfig,
) -> VisualQcResult:
    api_key = os.environ.get(config.glm_api_key_env)
    if not api_key:
        raise RuntimeError(f"GLM API key environment variable is not set: {config.glm_api_key_env}")

    content: list[dict[str, Any]] = [{"type": "text", "text": _build_prompt(image_path.name, measurement_rows, phenotype_rows)}]
    if config.glm_qc_use_original:
        content.append(_image_content(image_path, config.glm_qc_crop_max_side))
    if config.glm_qc_use_overlay:
        content.append(_image_content(overlay_path, config.glm_qc_crop_max_side))

    payload = {
        "model": config.glm_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
    }
    url = config.glm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=config.glm_qc_timeout_s) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]["content"]
        parsed = _parse_json_message(str(message))
    except Exception as exc:
        parsed = {
            "overall_status": "review",
            "flags": ["glm_request_failed"],
            "suspicious_instances": [],
            "missed_wells": [],
            "summary": f"GLM visual review failed: {type(exc).__name__}",
        }
    return normalize_visual_qc_result(
        parsed,
        image_name=image_path.name,
        overlay_path=overlay_path,
        model=config.glm_model,
    )


def should_review_image(
    phenotype_rows: list[dict[str, object]],
    scope: str,
) -> bool:
    if scope == "all":
        return True
    if scope == "summary":
        return True
    return any(str(row.get("qc_status", "")) == "review" for row in phenotype_rows)


def write_visual_qc_files(output_dir: str | Path, results: list[VisualQcResult]) -> None:
    if not results:
        return
    output = Path(output_dir)
    rows = [_csv_row(result) for result in results]
    write_csv(output / "visual_qc.csv", rows)
    (output / "visual_qc.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_visual_qc_result(
    result: dict[str, Any],
    *,
    image_name: str,
    overlay_path: Path,
    model: str,
) -> VisualQcResult:
    status = str(result.get("overall_status", "review")).lower()
    if status not in {"pass", "review"}:
        status = "review"
    flags = result.get("flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
    suspicious_instances = result.get("suspicious_instances", [])
    if not isinstance(suspicious_instances, list):
        suspicious_instances = []
    missed_wells = result.get("missed_wells", [])
    if not isinstance(missed_wells, list):
        missed_wells = [str(missed_wells)]
    return {
        "image_name": image_name,
        "overlay_path": str(overlay_path),
        "glm_model": model,
        "overall_status": status,
        "flags": [str(flag) for flag in flags],
        "suspicious_instances": suspicious_instances,
        "missed_wells": [str(well) for well in missed_wells],
        "summary": str(result.get("summary", "")),
        "raw": result,
    }


def _build_prompt(
    image_name: str,
    measurement_rows: list[dict[str, object]],
    phenotype_rows: list[dict[str, object]],
) -> str:
    detected = len(measurement_rows)
    review_wells = [str(row.get("well")) for row in phenotype_rows if str(row.get("qc_status", "")) == "review"][:30]
    context = {
        "image_name": image_name,
        "detected_instances": detected,
        "review_wells_sample": review_wells,
    }
    return (
        "你是抑菌圈图像质控助手。请对原图和/或带彩色 mask 编号的 overlay 做视觉复核，"
        "判断当前自动标注是否存在明显漏检、明显过分割、明显欠分割、mask 偏离孔中心、边界不清、图像质量问题。"
        "不要修改数值，不要给出新的 mask，只做 review-only 质控。"
        "如果整体可接受，overall_status 填 pass；如果需要人工复核，填 review。"
        "只返回严格 JSON，不要使用 Markdown 代码块。JSON schema 必须是："
        "{\"image_name\":string,\"overall_status\":\"pass|review\",\"flags\":[string],"
        "\"suspicious_instances\":[{\"instance_id\":number|string,\"well\":string,\"status\":\"review\",\"issue\":string,\"reason\":string}],"
        "\"missed_wells\":[string],\"summary\":string}。"
        f"\n结构化上下文：{json.dumps(context, ensure_ascii=False)}"
    )


def _image_content(path: Path, max_side: int) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{_encode_resized_jpeg(path, max_side)}",
        },
    }


def _encode_resized_jpeg(path: Path, max_side: int) -> str:
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return base64.b64encode(data).decode("utf-8")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side > 0:
        scale = max_side / float(longest)
        image = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return base64.b64encode(data).decode("utf-8")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _parse_json_message(message: str) -> dict[str, Any]:
    text = message.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"overall_status": "review", "flags": ["glm_non_json_response"], "summary": text[:500]}
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        return {"overall_status": "review", "flags": ["glm_invalid_json_shape"], "summary": str(parsed)[:500]}
    return parsed


def _csv_row(result: VisualQcResult) -> dict[str, object]:
    return {
        "image_name": result.get("image_name"),
        "overlay_path": result.get("overlay_path"),
        "glm_model": result.get("glm_model"),
        "overall_status": result.get("overall_status"),
        "flags": ";".join(str(flag) for flag in result.get("flags", [])),
        "suspicious_instances": json.dumps(result.get("suspicious_instances", []), ensure_ascii=False),
        "missed_wells": ";".join(str(well) for well in result.get("missed_wells", [])),
        "summary": result.get("summary"),
    }

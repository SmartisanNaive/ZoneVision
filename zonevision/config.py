from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ZoneVisionConfig:
    detector: str = "yolo"
    yolo_model: str = "yolo26n.pt"
    yolo_conf: float = 0.2
    use_yolo_seg: bool = False
    enable_yolo: bool = True
    rfdetr_model: str | None = None
    rfdetr_variant: str = "seg-small"
    rfdetr_conf: float = 0.3
    enable_rfdetr: bool = False
    rfdetr_max_detections: int = 120
    sam3_checkpoint: str = "ZoneVision/models/sam3.pt"
    enable_sam3: bool = True
    sam3_policy: str = "auto"
    device: str = "auto"
    imgsz: int = 640
    batch_size: int = 1
    roi_expand_ratio: float = 0.15
    sam3_roi_max_side: int = 512
    sam_mask_merge_policy: str = "replace"
    well_pitch_mm: float = 9.0
    min_zone_radius_mm: float = 1.5
    max_zone_radius_mm: float = 10.0
    bootstrap_intensity_delta: float = 7.0
    draw_measurements: bool = True
    fallback_to_bootstrap: bool = True
    glm_qc_enabled: bool = False
    glm_model: str = "glm-5v-turbo"
    glm_api_key_env: str = "GLM_API_KEY"
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    glm_qc_scope: str = "review"
    glm_qc_max_images: int = 20
    glm_qc_timeout_s: float = 30.0
    glm_qc_crop_max_side: int = 1024
    glm_qc_use_overlay: bool = True
    glm_qc_use_original: bool = True
    glm_qc_action: str = "flag_only"
    save_debug: bool = False
    max_images: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ZoneVisionConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def merge(self, overrides: dict[str, Any]) -> "ZoneVisionConfig":
        values = asdict(self)
        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
        return ZoneVisionConfig(**values)

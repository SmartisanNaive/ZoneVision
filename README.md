# ZoneVision

Automated inhibition-zone detection and measurement for antibiotic susceptibility testing (AST) plate photographs.

## Overview

ZoneVision is a cascade vision pipeline that detects and quantitatively measures inhibition zones (antibiotic halos) on 96-well plate color photographs. It combines object detection and instance segmentation to produce per-well phenotypic measurements (diameter, area) suitable for downstream biological analysis.

**Pipeline stages:**

1. **Plate geometry estimation** — YOLO26n + Hough Circles detect the 96-well grid and estimate pixel/mm scale
2. **Zone segmentation** — RF-DETR-Seg-Small performs end-to-end instance segmentation of inhibition zones
3. **Optional mask refinement** — SAM3 refines zone boundaries within detected ROIs
4. **Physical calibration** — Pixel measurements are converted to millimeters using the 9.0 mm well pitch
5. **Output** — CSV/JSON with per-well diameters, areas, and QC flags, plus overlay visualizations

## Results

| Metric | Value |
|--------|-------|
| F1 Score | 0.952 |
| Precision | 0.973 |
| Recall | 0.931 |
| Mean IoU | 0.896 |
| Diameter MAE | 0.234 mm (3.08%) |
| Pearson r | 0.973 |

Evaluated on 11 plate photos with 233 manually annotated zones.

## Installation

```bash
git clone https://github.com/SmartisanNaive/zonevision.git
cd zonevision
pip install -e .
```

### Download model weights

Weights are hosted on [HuggingFace](https://huggingface.co/SmartisanNaive/zonevision):

```bash
# Option 1: huggingface-cli
huggingface-cli download SmartisanNaive/zonevision --local-dir weights/

# Option 2: manual download
mkdir -p weights
# Download from https://huggingface.co/SmartisanNaive/zonevision
# Place files in weights/:
#   weights/sam3.pt
#   weights/rfdetr_seg_small_best.pth
#   weights/yolo26n.pt
#   weights/yolo26n-seg.pt
```

## Quick Start

```bash
# Run inference on plate photos
python scripts/run_pipeline.py \
  --input examples/sample_data/images/ \
  --output outputs/ \
  --config configs/config.yaml \
  --detector rfdetr

# Results will be in outputs/:
#   measurements.csv   — per-well diameter and area
#   overlays/          — annotated overlay images
#   masks/             — binary zone masks
```

## Training

```bash
# Train RF-DETR on a COCO dataset
python scripts/train_rfdetr.py \
  --dataset data/coco_zone_384/ \
  --output results/rfdetr_training/ \
  --epochs 100

# Train YOLO segmentation model
python scripts/train_yolo.py \
  --data data/yolo_zone_clean/data.yaml \
  --epochs 100
```

## Evaluation

```bash
# Evaluate predictions against COCO ground truth
python scripts/evaluate_predictions.py \
  --coco data/coco_zone_clean/ \
  --predictions outputs/predictions.json

# Generate benchmark report (Markdown + CSV)
python scripts/generate_benchmark_report.py \
  --gt data/coco_zone_clean/ \
  --pred-csv outputs/measurements.csv \
  --output benchmarks/
```

## Project Structure

```
zonevision/
├── zonevision/            # Core Python package
│   ├── config.py          # Configuration dataclass
│   ├── pipeline.py        # Main pipeline orchestration
│   ├── plate.py           # Plate geometry estimation
│   ├── postprocess.py     # Measurement extraction & QC
│   ├── rfdetr_integration.py  # RF-DETR wrappers
│   ├── glm_qc.py          # GLM-5V visual quality control
│   ├── coco.py            # COCO dataset utilities
│   └── io_utils.py        # Image I/O helpers
├── scripts/               # Entry-point scripts
├── configs/               # Configuration files
├── examples/              # Sample data and notebooks
├── benchmarks/            # Evaluation reports
├── weights/               # Model weights (gitignored)
└── pyproject.toml         # Package metadata
```

## Configuration

Edit `configs/config.yaml` to customize:

- `detector`: `rfdetr` (recommended) or `yolo`
- `rfdetr_conf` / `yolo_conf`: Detection confidence thresholds
- `enable_sam3`: Enable SAM3 mask refinement
- `well_pitch_mm`: Physical well spacing (default 9.0 mm)
- `glm_qc_enabled`: Enable GLM-5V visual review

## Requirements

- Python 3.12
- PyTorch >= 2.4
- Ultralytics >= 8.3
- RF-DETR >= 1.6
- OpenCV >= 4.10
- See `pyproject.toml` for full dependencies

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use ZoneVision in your research, please cite:

```bibtex
@article{zonevision2026,
  title={Automated Inhibition-Zone Detection for Antibiotic Susceptibility Testing Using Cascade Vision},
  author={Your Name},
  journal={Chinese Journal of Biotechnology},
  year={2026}
}
```

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) — Real-time detection transformer
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO implementation
- [SAM](https://github.com/facebookresearch.com/sam2) — Segment Anything Model

# Panoptic Segmentation Evaluation Guide

This guide provides comprehensive instructions for running panoptic segmentation evaluations on different devices and datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Evaluation Script Usage](#evaluation-script-usage)
4. [Device-Specific Instructions](#device-specific-instructions)
   - [MacBook (Apple Silicon)](#macbook-apple-silicon)
   - [ZedBox (NVIDIA Jetson)](#zedbox-nvidia-jetson)
   - [NVIDIA Workstation/Server](#nvidia-workstationserver)
5. [Dataset Configurations](#dataset-configurations)
   - [COCO Panoptic](#coco-panoptic-dataset)
   - [VALID Dataset](#valid-dataset)
6. [Output and Metrics](#output-and-metrics)
7. [Full Evaluation Examples](#full-evaluation-examples)

---

## Overview

The evaluation script (`scripts/eval_subset.py`) runs panoptic segmentation inference and computes:

- **PQ (Panoptic Quality)**: Overall segmentation quality
- **SQ (Segmentation Quality)**: How well segments match ground truth
- **RQ (Recognition Quality)**: How well categories are recognized
- **Carbon Emissions**: CO2 emissions tracked via CodeCarbon
- **Performance Metrics**: FPS, total inference time

---

## Prerequisites

Before running evaluations, ensure you have:

1. **Installed dependencies** (see `installation-instructions.md`)
2. **Downloaded model checkpoints**:
   ```bash
   poetry run mim download mmdet \
     --config maskformer_r50_ms-16xb1-75e_coco \
     --dest external/checkpoints

   poetry run mim download mmdet \
     --config maskformer_swin-l-p4-w12_64xb1-ms-300e_coco \
     --dest external/checkpoints
   ```
3. **Prepared datasets** (COCO and/or VALID)

---

## Evaluation Script Usage

### Basic Syntax

```bash
poetry run python scripts/eval_subset.py <config> <checkpoint> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `config` | Yes | Path to model config file (`.py`) |
| `checkpoint` | Yes | Path to model weights (`.pth`) |
| `--n <N>` | No | Number of images to evaluate. Use `0` for full dataset. Default: `5` |
| `--device <name>` | No | Device name for logging (e.g., `macbook`, `zedbox`, `nvidia`). Default: `unknown` |
| `--work-dir <path>` | No | Output directory for logs and emissions data |
| `--dataset-label <name>` | No | Custom label for dataset in metrics CSV |

### Example

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 100 \
    --device macbook \
    --work-dir work_dirs/eval_coco_macbook
```

---

## Device-Specific Instructions

### MacBook (Apple Silicon)

MacBooks with M1/M2/M3/M4 chips use MPS (Metal Performance Shaders) for acceleration.

#### Quick Test (5 images)

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 5 \
    --device macbook \
    --work-dir work_dirs/coco_macbook_n5
```

#### Full COCO Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device macbook \
    --work-dir work_dirs/coco_macbook_full
```

#### VALID Dataset Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device macbook \
    --work-dir work_dirs/valid_macbook_full
```

---

### ZedBox (NVIDIA Jetson)

ZedBox uses NVIDIA Jetson with CUDA acceleration. Ensure CUDA is properly configured.

#### Set GPU

```bash
export CUDA_VISIBLE_DEVICES=0
```

#### Quick Test (5 images)

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 5 \
    --device zedbox \
    --work-dir work_dirs/coco_zedbox_n5
```

#### Full COCO Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device zedbox \
    --work-dir work_dirs/coco_zedbox_full
```

#### VALID Dataset Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device zedbox \
    --work-dir work_dirs/valid_zedbox_full
```

---

### NVIDIA Workstation/Server

For workstations with dedicated NVIDIA GPUs (RTX 4000, A100, etc.).

#### Select GPU (multi-GPU systems)

```bash
# Use first GPU
export CUDA_VISIBLE_DEVICES=0

# Or use second GPU
export CUDA_VISIBLE_DEVICES=1
```

#### Quick Test (5 images)

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 5 \
    --device nvidia \
    --work-dir work_dirs/coco_nvidia_n5
```

#### Full COCO Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device nvidia \
    --work-dir work_dirs/coco_nvidia_full
```

#### VALID Dataset Evaluation

```bash
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device nvidia \
    --work-dir work_dirs/valid_nvidia_full
```

---

## Dataset Configurations

### COCO Panoptic Dataset

Uses the standard COCO panoptic validation set (5000 images).

**Config file:** `external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py`

**Data path:** `data/coco/`

```
data/coco/
├── val2017/                    # Validation images
├── annotations/
│   ├── panoptic_val2017.json   # Panoptic annotations
│   └── panoptic_val2017/       # Panoptic masks (PNG)
```

### VALID Dataset

Uses the VALID aerial drone dataset converted to COCO format.

**Config file:** `external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py`

**Data path:** `data/valid/`

```
data/valid/
├── val/                                    # Validation images
├── annotations/
│   ├── panoptic_val.json                   # Original annotations
│   ├── panoptic_val_coco_format.json       # COCO-format annotations (for evaluation)
│   └── panoptic_val/                       # Panoptic masks (PNG)
```

**Note:** The VALID dataset uses `panoptic_val_coco_format.json` for evaluation, which contains only COCO-compatible categories (21 out of 31 categories).

---

## Output and Metrics

### Metrics CSV

Results are appended to `outputs/run_metrics.csv`:

| Column | Description |
|--------|-------------|
| `device` | Device name (macbook, zedbox, nvidia) |
| `experiment` | Work directory name |
| `model` | Config file name |
| `dataset` | Dataset type |
| `num_samples` | Number of images evaluated |
| `total_seconds` | Total inference time |
| `fps` | Frames per second |
| `emissions_kg` | Carbon emissions (kg CO2) |
| `PQ` | Panoptic Quality (0-100) |
| `SQ` | Segmentation Quality (0-100) |
| `RQ` | Recognition Quality (0-100) |

### Work Directory

Each evaluation creates a work directory with:

```
work_dirs/<experiment_name>/
├── <timestamp>/
│   └── <timestamp>.log     # Detailed log
├── emissions.csv           # CodeCarbon emissions data
└── powermetrics_log.txt    # Power consumption (macOS)
```

---

## Full Evaluation Examples

### Compare All Devices on COCO (Full Dataset)

Run these commands on each respective device:

**MacBook:**
```bash
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device macbook
```

**ZedBox:**
```bash
export CUDA_VISIBLE_DEVICES=0
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device zedbox
```

**NVIDIA Workstation:**
```bash
export CUDA_VISIBLE_DEVICES=0
poetry run python scripts/eval_subset.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device nvidia
```

### Compare All Devices on VALID (Full Dataset)

**MacBook:**
```bash
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device macbook
```

**ZedBox:**
```bash
export CUDA_VISIBLE_DEVICES=0
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device zedbox 
```

**NVIDIA Workstation:**
```bash
export CUDA_VISIBLE_DEVICES=0
poetry run python scripts/eval_subset.py \
    external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_coco_on_valid.py \
    external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
    --n 0 \
    --device nvidia
```

---

## Available Models

| Model | Config | Checkpoint | Size | Notes |
|-------|--------|------------|------|-------|
| MaskFormer R50 | `maskformer_r50_ms-16xb1-75e_coco.py` | `maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth` | ~170MB | Smaller, faster |
| MaskFormer Swin-L | `maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py` | `maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth` | ~840MB | Larger, more accurate |

---

## Tips

1. **Quick testing:** Use `--n 5` or `--n 10` for quick sanity checks before full runs
2. **Reproducibility:** Results are appended to CSV, making it easy to compare across runs
3. **Carbon tracking:** Check `work_dirs/<experiment>/emissions.csv` for detailed emissions data
4. **GPU selection:** Always set `CUDA_VISIBLE_DEVICES` on multi-GPU systems to ensure consistent results


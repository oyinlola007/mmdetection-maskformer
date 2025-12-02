Panoptic Segmentation – Green AI with MMDetection
=================================================

This repository contains experiments on **panoptic segmentation** with a focus on **green AI**: measuring and reducing the carbon footprint of modern segmentation models during training and inference. Models are evaluated on Cityscapes and COCO, then adapted to aerial imagery (iSAID), and deployment is targeted to an edge device (ZED Box).

Main Components
---------------

- `apps/`: Streamlit demo(s) for interactive panoptic inference.
- `configs/`: Run and experiment configuration files.
- `scripts/`: Utility scripts (e.g. evaluation subsets).
- `data/`: Dataset structure (COCO / iSAID).

Environment & Tools
-------------------

- Python dependency management via `poetry` (`pyproject.toml`, `poetry.lock`).
- Panoptic segmentation with MMDetection.
- Carbon tracking with CodeCarbon and CarbonTracker.

Quick Start (Workstation)
-------------------------

```bash
# Install dependencies
poetry install

# Install MMDetection and related tools (example)
poetry run mim install mmengine
poetry run mim install "mmcv==2.1.0"
poetry run mim install mmdet

# Example: run Streamlit panoptic demo (if enabled)
poetry run streamlit run apps/panoptic_inference_app.py
```

Datasets and checkpoints must be placed under `data/` and `external/checkpoints/` respectively, following the structure described in `data/README.md` and `steps.txt`. These large files are intentionally ignored by git so the public repo only contains code and lightweight metadata.

Detailed Setup & Experiment Steps
---------------------------------

The following steps document how to set up PyTorch with MPS, MMDetection, and energy/CO2 trackers on Apple Silicon.

```bash
# Go to project root
cd "/Users/oyinlola/Desktop/ESIGELEC/R&D/MMDetection"

# Install Poetry (required)
curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is on PATH (adjust if your installer uses a different location)
export PATH="$HOME/.local/bin:$PATH"

poetry env use python3.11

poetry run pip install --upgrade pip setuptools wheel

# Install dependencies from pyproject (package-mode=false avoids local package installation)
poetry install

poetry run mim install mmengine

poetry run mim install "mmcv==2.1.0"

poetry run mim install mmdet

# Sanity check and run
poetry run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Install COCO panoptic dependency required by PanopticFPN pipelines
poetry run pip install "git+https://github.com/cocodataset/panopticapi.git"

# Prepare external MMDetection configs
mkdir -p external/mmdet-configs/maskformer

mkdir -p external/mmdet-configs/_base_/datasets external/mmdet-configs/_base_/schedules external/mmdet-configs/_base_/models

curl -L -o external/mmdet-configs/_base_/datasets/coco_panoptic.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/datasets/coco_panoptic.py

curl -L -o external/mmdet-configs/_base_/default_runtime.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/default_runtime.py

curl -L -o external/mmdet-configs/_base_/schedules/schedule_50e.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/schedules/schedule_50e.py

curl -L -o external/mmdet-configs/_base_/models/mask2former_r50.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/models/mask2former_r50.py

# Download COCO MaskFormer checkpoints
poetry run mim download mmdet \
  --config maskformer_r50_ms-16xb1-75e_coco \
  --dest external/checkpoints

poetry run mim download mmdet \
  --config maskformer_swin-l-p4-w12_64xb1-ms-300e_coco \
  --dest external/checkpoints

# Run COCO panoptic mini-eval (PQ/SQ/RQ + AP bbox/segm + energy)
# use --n 0 to run all
poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py \
  external/checkpoints/maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth \
  --n 5

poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
  external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
  --n 5

poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_valid.py \
  external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
  --n 0 \
  --dataset-label valid_baseline
```


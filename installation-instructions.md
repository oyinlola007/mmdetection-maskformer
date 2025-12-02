# MMDetection Installation Guide

This guide documents the steps to set up PyTorch, MMDetection, and energy/CO2 trackers for panoptic segmentation experiments.

## Verified Compatible Versions (Dec 2025)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | |
| PyTorch | 2.1.2 | CUDA 12.1 for server, MPS for macOS |
| torchvision | 0.16.2 | |
| mmengine | 0.10.3 | **Pinned** - newer versions require PyTorch 2.2+ |
| mmcv | 2.1.0 | Pre-built wheel from OpenMMLab |
| mmdet | 3.3.0 | |
| transformers | 4.36.0 | **Pinned** - newer versions require PyTorch 2.2+ |
| numpy | 1.26.4 | |

> **Important:** The version pinning for `mmengine` and `transformers` is critical. Newer versions use `register_pytree_node` which only exists in PyTorch 2.2+.

---

## Section 1: Apple Silicon (macOS with MPS)

### Step 1: Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Step 2: Set up environment

```bash
cd "/path/to/MMDetection"
poetry env use python3.11
poetry run pip install --upgrade pip setuptools wheel
```

### Step 3: Install dependencies

```bash
# Install base dependencies from pyproject.toml
poetry install

# Install MMEngine (pinned version)
poetry run pip install mmengine==0.10.3

# Install mmcv
poetry run mim install "mmcv==2.1.0"

# Install MMDetection
poetry run mim install mmdet==3.3.0

# Install transformers (pinned version)
poetry run pip install transformers==4.36.0

# Install COCO panoptic API
poetry run pip install "git+https://github.com/cocodataset/panopticapi.git"
```

### Step 4: Verify installation

```bash
poetry run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Section 2: Ubuntu Server with CUDA

**Tested on:** Ubuntu with 2x NVIDIA RTX 4000 Ada Generation (20GB each), CUDA 12.1

### Step 1: Set up conda environment

```bash
conda create -n mmdetection python=3.11 -y
conda activate mmdetection
```

### Step 2: Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Step 3: Clean up existing installations (if any)

```bash
poetry run pip uninstall torch torchvision mmcv mmcv-full mmdet mmengine -y
poetry run pip cache purge
```

### Step 4: Install PyTorch with CUDA support

```bash
poetry run pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Verify PyTorch and CUDA

```bash
poetry run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

Expected output:
```
PyTorch: 2.1.2+cu121
CUDA available: True
CUDA version: 12.1
```

### Step 6: Install MMEngine (pinned version)

```bash
poetry run pip install mmengine==0.10.3
```

### Step 7: Install mmcv from pre-built wheels

```bash
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### Step 8: Verify mmcv CUDA extensions

```bash
poetry run python -c "from mmcv import _ext; print('MMCV CUDA extensions found')"
```

### Step 9: Install MMDetection

```bash
poetry run pip install mmdet==3.3.0
```

### Step 10: Install transformers (pinned version)

```bash
poetry run pip install transformers==4.36.0
```

### Step 11: Install COCO panoptic API

```bash
poetry run pip install "git+https://github.com/cocodataset/panopticapi.git"
```

### Step 12: Final verification

```bash
poetry run python -c "
import torch
import mmcv
import mmengine
import mmdet

print('=' * 50)
print('All packages imported successfully!')
print('=' * 50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmengine: {mmengine.__version__}')
print(f'mmdet: {mmdet.__version__}')
print('=' * 50)
"
```

Expected output:
```
==================================================
All packages imported successfully!
==================================================
PyTorch: 2.1.2+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 2
GPU name: NVIDIA RTX 4000 Ada Generation
mmcv: 2.1.0
mmengine: 0.10.3
mmdet: 3.3.0
==================================================
```

---

## Section 3: Download Configs and Checkpoints

### Create directories

```bash
mkdir -p external/mmdet-configs/maskformer
mkdir -p external/mmdet-configs/_base_/datasets
mkdir -p external/mmdet-configs/_base_/schedules
mkdir -p external/mmdet-configs/_base_/models
```

### Download base configs

```bash
curl -L -o external/mmdet-configs/_base_/datasets/coco_panoptic.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/datasets/coco_panoptic.py

curl -L -o external/mmdet-configs/_base_/default_runtime.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/default_runtime.py

curl -L -o external/mmdet-configs/_base_/schedules/schedule_50e.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/schedules/schedule_50e.py

curl -L -o external/mmdet-configs/_base_/models/mask2former_r50.py \
  https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/_base_/models/mask2former_r50.py
```

### Download model checkpoints

```bash
poetry run mim download mmdet \
  --config maskformer_r50_ms-16xb1-75e_coco \
  --dest external/checkpoints

poetry run mim download mmdet \
  --config maskformer_swin-l-p4-w12_64xb1-ms-300e_coco \
  --dest external/checkpoints
```

---

## Section 4: Run Evaluation

### Select GPU (server only)

```bash
export CUDA_VISIBLE_DEVICES=0
```

### MaskFormer R50 (smaller model)

```bash
poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py \
  external/checkpoints/maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth \
  --n 5
```

### MaskFormer Swin-L (larger model)

```bash
poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py \
  external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
  --n 5
```

### Full validation run

```bash
poetry run python scripts/eval_subset.py \
  external/mmdet-configs/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_valid.py \
  external/checkpoints/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth \
  --n 0 \
  --dataset-label valid_baseline
```

---

## Troubleshooting

### ERROR: `register_pytree_node` AttributeError

**Cause:** MMEngine or transformers version is too new for PyTorch 2.1.x

**Fix:**
```bash
poetry run pip install mmengine==0.10.3 transformers==4.36.0
```

### ERROR: `undefined symbol` with mmcv

**Cause:** mmcv was built from source instead of using pre-built wheel

**Fix:**
```bash
poetry run pip uninstall mmcv -y
poetry run pip cache purge
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### ERROR: `No module named 'mmcv._ext'`

**Cause:** mmcv installed without CUDA extensions

**Fix:** Same as above - reinstall from pre-built wheels

### ERROR: `mmcv incompatible` (requires mmcv < 2.2.0)

**Cause:** mmcv 2.2.0 installed, but MMDetection 3.3.0 requires < 2.2.0

**Fix:**
```bash
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### ERROR: mmcv wheel not found (builds from source)

**Cause:** PyTorch version doesn't match the wheel index

**Fix:** Ensure PyTorch version is exactly 2.1.x for the torch2.1 index:
```bash
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

---

## Version Compatibility Notes

The compatibility chain that led to these specific versions:

1. **MMDetection 3.3.0** requires `mmcv >= 2.0.0rc4, < 2.2.0`
2. **mmcv 2.1.0** pre-built wheels are available for PyTorch 2.1.x + CUDA 12.1
3. **mmcv 2.2.0** pre-built wheels are available for PyTorch 2.2.x + CUDA 12.1, but rejected by MMDetection 3.3.0
4. **MMEngine 0.10.7+** uses `register_pytree_node` which requires PyTorch 2.2+
5. **transformers 4.37+** uses `register_pytree_node` which requires PyTorch 2.2+
6. **PyTorch 2.1.x** only has `_register_pytree_node` (private API)

**Solution:** Pin MMEngine to 0.10.3 and transformers to 4.36.0 to maintain compatibility with PyTorch 2.1.2.


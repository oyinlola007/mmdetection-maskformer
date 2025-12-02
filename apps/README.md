# Apps

This directory contains lightweight interactive apps (Streamlit) for running inference and demos locally.

## Panoptic Inference App

Run a simple Streamlit app to perform panoptic segmentation on images from file upload or webcam, using MMDetection MaskFormer models.

Usage:

```bash
poetry install
poetry run mim install mmengine
poetry run mim install "mmcv==2.1.0"
poetry run mim install mmdet

poetry run streamlit run apps/panoptic_inference_app.py
```

The app lets you choose between:
- `maskformer_r50_ms-16xb1-75e_coco`
- `maskformer_swin-l-p4-w12_64xb1-ms-300e_coco`

It auto-selects `mps` if available on Apple Silicon, else falls back to CPU.


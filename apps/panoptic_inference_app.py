import io
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st


def get_device() -> str:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _restart_app():
    try:
        # Clear Streamlit session state and cached resources
        st.session_state.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    # Rerun the script to reset all UI state
    st.rerun()


@st.cache_resource(show_spinner=False)
def load_inferencer(config_path: str, checkpoint_path: str, device: str):
    from mmdet.apis import DetInferencer

    # Use config + checkpoint to avoid ambiguity; device can be 'mps' or 'cpu'
    return DetInferencer(model=config_path, weights=checkpoint_path, device=device)


def image_to_ndarray(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    return np.array(image)


def run_inference(
    inferencer,
    img_np: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        # return_vis=True returns BGR ndarray(s) under key 'visualization'
        result = inferencer(
            img_np,
            return_vis=True,
            # Do not pass vis_out_dir/pred_out_dir; not supported in current MMDetection API
        )
        vis_list = result.get("visualization", None)
        if vis_list is None or len(vis_list) == 0:
            return None, "No visualization returned by inferencer."
        vis_bgr = vis_list[0]
        # Convert BGR->RGB for display
        vis_rgb = vis_bgr[:, :, ::-1]
        return vis_rgb, None
    except RuntimeError as e:
        error_msg = str(e)
        # Check for MPS-specific errors
        if "MPS" in error_msg or "mps" in error_msg.lower():
            return (
                None,
                f"MPS backend error: {error_msg}. Try using CPU device or a smaller model/image.",
            )
        return None, f"Runtime error: {error_msg}"
    except Exception as e:
        return None, f"Inference error: {str(e)}"


def main():
    st.set_page_config(page_title="Panoptic Segmentation – MMDetection", layout="wide")
    st.title("Panoptic Segmentation Demo (MaskFormer)")
    st.caption(
        "Select a MaskFormer model, upload an image or use webcam, and view results side-by-side."
    )

    # Busy flag to disable inputs while inference runs
    if "busy" not in st.session_state:
        st.session_state["busy"] = False
    is_busy = bool(st.session_state.get("busy", False))

    device = get_device()
    with st.sidebar:
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose a model",
            options=(
                "MaskFormer R50 (75e COCO)",
                "MaskFormer Swin-L (300e COCO)",
            ),
            disabled=is_busy,
        )

        if model_choice == "MaskFormer R50 (75e COCO)":
            config_path = (
                "external/mmdet-configs/maskformer/"
                "maskformer_r50_ms-16xb1-75e_coco.py"
            )
            checkpoint_path = (
                "external/checkpoints/"
                "maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth"
            )
        else:
            config_path = (
                "external/mmdet-configs/maskformer/"
                "maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py"
            )
            checkpoint_path = (
                "external/checkpoints/"
                "maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth"
            )

        st.markdown("**Device**: %s" % device)
        st.markdown("<small>Config:</small> `%s`" % config_path, unsafe_allow_html=True)
        st.markdown(
            "<small>Checkpoint:</small> `%s`" % checkpoint_path, unsafe_allow_html=True
        )

        st.subheader("Input Source")
        input_source = st.radio(
            "Select input type",
            options=("Upload Image", "Webcam"),
            index=0,
            disabled=is_busy,
        )

        if input_source == "Webcam":
            cam_size_label = st.selectbox(
                "Camera preview size",
                options=["Small", "Medium", "Large"],
                index=1,
                disabled=is_busy,
            )
        else:
            cam_size_label = None

        st.divider()
        if st.button("Restart app", width="stretch", disabled=is_busy):
            _restart_app()

    st.subheader("Input")
    uploaded = None
    captured = None

    if input_source == "Upload Image":
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
            disabled=is_busy,
        )
    else:
        # Adjust camera preview height via CSS for a smaller widget
        size_map = {"Small": 240, "Medium": 360, "Large": 480}
        cam_h = size_map.get(cam_size_label or "Medium", 360)
        st.markdown(
            f"""
            <style>
            div[data-testid="stCamera"] > div {{
                height: {cam_h}px !important;
                max-height: {cam_h}px !important;
            }}
            div[data-testid="stCamera"] video {{
                height: {cam_h}px !important;
                max-height: {cam_h}px !important;
            }}
            div[data-testid="stCamera"] canvas {{
                height: {cam_h}px !important;
                max-height: {cam_h}px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        captured = st.camera_input("Take a picture with your webcam", disabled=is_busy)

    input_image: Optional[Image.Image] = None
    if uploaded is not None:
        input_image = Image.open(uploaded)
    elif captured is not None:
        # st.camera_input returns an UploadedFile; read to PIL
        input_image = Image.open(io.BytesIO(captured.getvalue()))

    if input_image is None:
        st.info("Upload an image or capture from webcam to run inference.")
        return

    # Show original image
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(input_image, width="stretch")

    # Load model lazily and run inference; disable inputs while running
    st.session_state["busy"] = True
    try:
        with st.spinner("Loading model and running inference..."):
            inferencer = load_inferencer(config_path, checkpoint_path, device)
            img_np = image_to_ndarray(input_image)
            vis_rgb, err = run_inference(inferencer, img_np)
            # If MPS fails, try CPU as fallback
            if err is not None and "MPS" in err and device == "mps":
                st.warning("MPS backend failed. Retrying with CPU...")
                # Clear the cached inferencer and reload with CPU
                load_inferencer.clear()
                inferencer = load_inferencer(config_path, checkpoint_path, "cpu")
                vis_rgb, err = run_inference(inferencer, img_np)
    finally:
        st.session_state["busy"] = False

    with col2:
        st.markdown("**Panoptic Segmentation**")
        if err is not None:
            st.error(f"Inference error: {err}")
        elif vis_rgb is None:
            st.warning("No visualization output.")
        else:
            st.image(vis_rgb, width="stretch")


if __name__ == "__main__":
    main()

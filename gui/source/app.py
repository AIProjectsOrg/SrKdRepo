# app.py
import io
import os
import time
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort


# ---------------- Utilities ----------------
def pad_to_multiple(img_bgr: np.ndarray, m: int):
    """Reflect-pad H,W to multiple of m. Return padded image + (ph, pw)."""
    if m <= 1:
        return img_bgr, 0, 0
    h, w = img_bgr.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return img_bgr, ph, pw


def read_image_file(upload) -> np.ndarray:
    """Read an uploaded image file (Streamlit UploadedFile) into BGR uint8."""
    data = np.frombuffer(upload.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def file_fingerprint(path: str) -> str:
    """Stable fingerprint for caching the session by file identity."""
    p = Path(path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    key = f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    return hashlib.sha256(key.encode()).hexdigest()


def get_io_names(sess: ort.InferenceSession):
    """Return (input_name, output_name) for first input/output."""
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    if not inps or not outs:
        raise RuntimeError("ONNX model has no inputs/outputs.")
    return inps[0].name, outs[0].name


@st.cache_resource(show_spinner=False)
def load_session_from_path(onnx_path: str):
    """Cache ORT session (CPU EP for maximum Windows stability)."""
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner=False)
def load_session_from_bytes(onnx_bytes: bytes):
    """Cache ORT session from uploaded bytes (via temp file)."""
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx").name
    with open(path, "wb") as f:
        f.write(onnx_bytes)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]), path


def run_sr_session(
    sess: ort.InferenceSession,
    lr_bgr: np.ndarray,
    window_size: int = 8,
    normalize: str = "0to1",  # or "-1to1"
):
    """
    Run SR model:
    - reflect-pad to multiple of window_size
    - convert BGR->RGB and normalize
    - NCHW float32
    - forward
    - crop padding * scale
    - return BGR uint8, latency (s), inferred scale
    Mirrors the provided PyTorch code path.
    """
    t0 = time.time()

    # 1) Pad LR to multiple of window_size (reflect)
    lr_bgr_pad, ph, pw = pad_to_multiple(lr_bgr, window_size)

    # 2) Convert to RGB + normalize
    rgb = cv2.cvtColor(lr_bgr_pad, cv2.COLOR_BGR2RGB).astype(np.float32)
    if normalize == "0to1":
        rgb /= 255.0
    elif normalize == "-1to1":
        rgb = (rgb / 127.5) - 1.0
    else:
        raise ValueError("normalize must be '0to1' or '-1to1'")

    # 3) NCHW float32
    inp = np.transpose(rgb, (2, 0, 1))[None, ...]

    # 4) Input/output names
    in_name, out_name = get_io_names(sess)

    # 5) Forward
    out = sess.run([out_name], {in_name: inp})[0]  # 1x3xHxW or CHW if squeezed by exporter
    if out.ndim == 4:
        out = out[0]  # CHW
    # 6) HWC RGB
    sr = np.transpose(out, (1, 2, 0))

    # 7) Infer scale (e.g., 4 for x4)
    scale = int(round(sr.shape[0] / lr_bgr_pad.shape[0]))
    # 8) Crop padding on HR
    if ph or pw:
        H, W = sr.shape[:2]
        sr = sr[: H - ph * scale, : W - pw * scale, :]

    # 9) Denormalize + clamp
    if normalize == "0to1":
        sr = np.clip(sr, 0.0, 1.0)
        sr = (sr * 255.0).round().astype(np.uint8)
    else:  # "-1to1"
        sr = (np.clip(sr, -1.0, 1.0) + 1.0) * 127.5
        sr = np.clip(sr, 0, 255).round().astype(np.uint8)

    # 10) RGB->BGR for display/save
    sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    t1 = time.time()
    return sr_bgr, (t1 - t0), scale


# ---------------- UI ----------------
st.set_page_config(page_title="SwinIR/MicroSR (ONNX Runtime) • Streamlit", layout="wide")
st.title("🖼️ Super-Resolution (ONNX Runtime, ONNX-agnostic)")

with st.sidebar:
    st.header("Model")
    model_mode = st.radio("Model source", ["Upload .onnx", "Local path (.onnx)"], index=0)

    sess = None
    tmp_model_path = None
    if model_mode == "Upload .onnx":
        up_model = st.file_uploader("Upload ONNX model", type=["onnx"])
        if up_model is not None:
            try:
                # Use bytes for caching; keep temp path around for debugging if needed
                sess, tmp_model_path = load_session_from_bytes(up_model.getvalue())
                st.success("Model loaded (CPUExecutionProvider).")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
    else:
        model_path = st.text_input("Absolute path to ONNX model", value="")
        if model_path:
            try:
                _ = file_fingerprint(model_path)  # ensure file exists; load_session caches by path
                sess = load_session_from_path(model_path)
                st.success("Model loaded (CPUExecutionProvider).")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.markdown("---")
    st.header("Inference options")
    window_size = st.number_input(
        "Window size (multiple for padding; 8 for SwinIR, 16 for MicroSR example)",
        min_value=1, max_value=64, value=8, step=1,
    )
    normalize = st.selectbox(
        "Input normalization",
        options=["0to1", "-1to1"],
        index=0,
        help="Most SwinIR/MicroSR exports expect [0,1]; switch to [-1,1] if your model was exported that way."
    )

    st.markdown("---")
    st.header("Input image")
    img_file = st.file_uploader("Upload LR image", type=["png", "jpg", "jpeg", "bmp"])

    run_btn = st.button("🚀 Run Super-Resolution", type="primary", width='stretch')

col1, col2 = st.columns(2)

if run_btn:
    if sess is None:
        st.error("Please load an ONNX model first.")
    elif img_file is None:
        st.error("Please upload an LR image.")
    else:
        lr_bgr = read_image_file(img_file)
        if lr_bgr is None:
            st.error("Failed to read the uploaded image.")
        else:
            try:
                with st.spinner("🚀 Running super-resolution..."):
                    sr_bgr, latency, scale = run_sr_session(
                        sess, lr_bgr, window_size=window_size, normalize=normalize
                    )
                st.success(f"Inference complete in {latency:.2f}s (×{scale}).")

                with col1:
                    st.subheader("LR (input)")
                    st.image(cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                    st.caption(f"Resolution: {lr_bgr.shape[1]} × {lr_bgr.shape[0]}")

                with col2:
                    st.subheader(f"SR (×{scale})")
                    st.image(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                    st.caption(f"Resolution: {sr_bgr.shape[1]} × {sr_bgr.shape[0]} • Time: {latency:.2f}s")

                st.success("Inference complete.")
            except Exception as e:
                st.error(f"Inference failed: {e}")

with st.expander("ℹ️ Notes"):
    st.markdown(
        """
- This app mirrors the provided PyTorch inference: reflect-pad to a multiple of **window_size**, forward, then crop the ×**scale** padding.
- **Model-agnostic**: works for SwinIR, MicroSR, or other SR models as long as the ONNX expects NCHW float32 and standard normalization.
- **Input/Output names** are auto-detected from the ONNX graph (first input/output).
- **Scale** is inferred from output vs input height (so ×2/×3/×4 are all fine).
- Use **window size = 8** for typical SwinIR; **16** for your MicroSR example.
- Keep CPUExecutionProvider for maximum stability on Windows. (GPU can be added later.)
        """
    )

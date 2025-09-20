# app_gpu.py
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


def pick_providers(mode: str) -> list[str]:
    """
    Decide provider order based on user choice and availability.
    - Auto: prefer CUDA, then DirectML, then CPU
    - CUDA: CUDA then CPU
    - DirectML: DirectML then CPU
    - CPU: CPU only
    """
    avail = set(ort.get_available_providers())
    order = []

    def has_cuda(): return "CUDAExecutionProvider" in avail
    def has_dml(): return ("DmlExecutionProvider" in avail) or ("DirectMLExecutionProvider" in avail)
    dml_name = "DmlExecutionProvider" if "DmlExecutionProvider" in avail else (
        "DirectMLExecutionProvider" if "DirectMLExecutionProvider" in avail else None
    )

    if mode == "Auto":
        if has_cuda(): order.append("CUDAExecutionProvider")
        elif has_dml() and dml_name: order.append(dml_name)
        order.append("CPUExecutionProvider")
    elif mode == "CUDA":
        if has_cuda(): order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else: order = ["CPUExecutionProvider"]
    elif mode == "DirectML":
        if has_dml() and dml_name: order = [dml_name, "CPUExecutionProvider"]
        else: order = ["CPUExecutionProvider"]
    else:  # CPU
        order = ["CPUExecutionProvider"]

    return order


# ---- Session loaders (cache by model identity + providers) ----
@st.cache_resource(show_spinner=False)
def load_session_from_path(onnx_path: str, providers: tuple[str, ...]):
    """Cache ORT session for a given path+providers."""
    return ort.InferenceSession(onnx_path, providers=list(providers))


@st.cache_resource(show_spinner=False)
def load_session_from_bytes(onnx_bytes: bytes, providers: tuple[str, ...]):
    """Cache ORT session from uploaded bytes (via temp file) with providers."""
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx").name
    with open(path, "wb") as f:
        f.write(onnx_bytes)
    return ort.InferenceSession(path, providers=list(providers)), path


def run_sr_session(
    sess: ort.InferenceSession,
    lr_bgr: np.ndarray,
    window_size: int = 8,
    normalize: str = "0to1",  # or "-1to1"
):
    """Run SR inference with padding, normalization, cropping, and postproc."""
    t0 = time.time()

    lr_bgr_pad, ph, pw = pad_to_multiple(lr_bgr, window_size)
    rgb = cv2.cvtColor(lr_bgr_pad, cv2.COLOR_BGR2RGB).astype(np.float32)
    if normalize == "0to1":
        rgb /= 255.0
    elif normalize == "-1to1":
        rgb = (rgb / 127.5) - 1.0

    inp = np.transpose(rgb, (2, 0, 1))[None, ...]
    in_name, out_name = get_io_names(sess)

    out = sess.run([out_name], {in_name: inp})[0]
    if out.ndim == 4:
        out = out[0]
    sr = np.transpose(out, (1, 2, 0))

    scale = int(round(sr.shape[0] / lr_bgr_pad.shape[0]))
    if ph or pw:
        H, W = sr.shape[:2]
        sr = sr[: H - ph * scale, : W - pw * scale, :]

    if normalize == "0to1":
        sr = np.clip(sr, 0.0, 1.0)
        sr = (sr * 255.0).round().astype(np.uint8)
    else:
        sr = (np.clip(sr, -1.0, 1.0) + 1.0) * 127.5
        sr = np.clip(sr, 0, 255).round().astype(np.uint8)

    sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    return sr_bgr, (time.time() - t0), scale


# ---------------- UI ----------------
st.set_page_config(page_title="SwinIR/MicroSR (ONNX Runtime GPU) • Streamlit", layout="wide")
st.title("🖼️ Super-Resolution (ONNX Runtime • GPU/CPU)")

with st.sidebar:
    st.header("Model")
    model_mode = st.radio("Model source", ["Upload .onnx", "Local path (.onnx)"], index=0)

    st.markdown("---")
    st.header("Execution Provider")
    avail_eps = ort.get_available_providers()
    st.caption(f"Available providers: `{avail_eps}`")
    ep_mode = st.selectbox("Select provider", options=["Auto", "CUDA", "DirectML", "CPU"], index=0)

    st.markdown("---")
    st.header("Inference options")
    window_size = st.number_input("Window size (8 for SwinIR, 16 for MicroSR)", min_value=1, max_value=64, value=8)
    normalize = st.selectbox("Input normalization", options=["0to1", "-1to1"], index=0)

    st.markdown("---")
    st.header("Input image")
    img_file = st.file_uploader("Upload LR image", type=["png", "jpg", "jpeg", "bmp"])

    run_btn = st.button("🚀 Run Super-Resolution", type="primary")

sess = None
providers = pick_providers(ep_mode)
providers_t = tuple(providers)

if model_mode == "Upload .onnx":
    up_model = st.sidebar.file_uploader("Upload ONNX model", type=["onnx"], key="model_uploader")
    if up_model is not None:
        sess, _ = load_session_from_bytes(up_model.getvalue(), providers_t)
        st.sidebar.success(f"Model loaded ({providers[0]}).")
else:
    model_path = st.sidebar.text_input("Absolute path to ONNX model", value="")
    if model_path:
        sess = load_session_from_path(model_path, providers_t)
        st.sidebar.success(f"Model loaded ({providers[0]}).")

col1, col2 = st.columns(2)

if run_btn:
    if sess is None:
        st.error("Please load an ONNX model first.")
    elif img_file is None:
        st.error("Please upload an LR image.")
    else:
        lr_bgr = read_image_file(img_file)
        if lr_bgr is not None:
            sr_bgr, latency, scale = run_sr_session(sess, lr_bgr, window_size, normalize)
            st.success(f"Inference complete in {latency:.2f}s (×{scale}) using {providers[0]}.")

            with col1:
                st.subheader("LR (input)")
                st.image(cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB), channels="RGB", width=512)
                st.caption(f"Resolution: {lr_bgr.shape[1]} × {lr_bgr.shape[0]}")

            with col2:
                st.subheader(f"SR (×{scale})")
                st.image(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB), channels="RGB", width=512)
                st.caption(f"Resolution: {sr_bgr.shape[1]} × {sr_bgr.shape[0]} • Time: {latency:.2f}s")

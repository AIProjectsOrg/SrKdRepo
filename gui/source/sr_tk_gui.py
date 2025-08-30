# sr_tk_gui.py
import os
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk


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


def get_io_names(sess: ort.InferenceSession):
    """Return (input_name, output_name) for first input/output."""
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    if not inps or not outs:
        raise RuntimeError("ONNX model has no inputs/outputs.")
    return inps[0].name, outs[0].name


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def fit_image_for_preview(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    if pil_img.width <= max_w and pil_img.height <= max_h:
        return pil_img
    pil_img = pil_img.copy()
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return pil_img


# ---------------- App ----------------
class SRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SwinIR / MicroSR • ONNX (Tkinter)")
        self.geometry("1200x700")

        # State
        self.onnx_path: str | None = None
        self.sess: ort.InferenceSession | None = None
        self.lr_bgr: np.ndarray | None = None
        self.sr_bgr: np.ndarray | None = None
        self.io_names: tuple[str, str] | None = None

        # UI Vars
        self.window_size_var = tk.IntVar(value=8)       # 8 (SwinIR) / 16 (MicroSR)
        self.normalize_var = tk.StringVar(value="0to1") # "0to1" or "-1to1"
        self.status_var = tk.StringVar(value="Load an ONNX model and an LR image to begin.")

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Top toolbar
        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        for i in range(8):
            toolbar.columnconfigure(i, weight=0)
        toolbar.columnconfigure(8, weight=1)

        ttk.Button(toolbar, text="Load ONNX", command=self.load_model).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="Load LR Image", command=self.load_image).grid(row=0, column=1, padx=4)

        ttk.Label(toolbar, text="Window size:").grid(row=0, column=2, padx=(16, 4))
        self.ws_spin = ttk.Spinbox(toolbar, from_=1, to=64, width=5, textvariable=self.window_size_var)
        self.ws_spin.grid(row=0, column=3, padx=4)

        ttk.Label(toolbar, text="Normalize:").grid(row=0, column=4, padx=(16, 4))
        self.norm_combo = ttk.Combobox(toolbar, width=7, state="readonly",
                                       values=["0to1", "-1to1"], textvariable=self.normalize_var)
        self.norm_combo.grid(row=0, column=5, padx=4)

        self.run_btn = ttk.Button(toolbar, text="Run Inference", command=self.run_inference, state="disabled")
        self.run_btn.grid(row=0, column=6, padx=(16, 4))

        self.save_btn = ttk.Button(toolbar, text="Save SR", command=self.save_sr, state="disabled")
        self.save_btn.grid(row=0, column=7, padx=4)

        # Status + progress
        status_frame = ttk.Frame(self, padding=(8, 0, 8, 8))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_lbl = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_lbl.grid(row=0, column=0, sticky="ew")
        self.pbar = ttk.Progressbar(status_frame, mode="indeterminate", length=200)
        self.pbar.grid(row=0, column=1, padx=8)

        # Split previews
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        for f in (left, right):
            f.rowconfigure(1, weight=1)
            f.columnconfigure(0, weight=1)

        ttk.Label(left, text="LR (input)").grid(row=0, column=0, sticky="w")
        self.lr_canvas = tk.Canvas(left, bg="#FFFFFF", highlightthickness=1, highlightbackground="#CCCCCC")
        self.lr_canvas.grid(row=1, column=0, sticky="nsew")
        self.lr_meta = ttk.Label(left, text="–")
        self.lr_meta.grid(row=2, column=0, sticky="w", pady=(4,0))

        ttk.Label(right, text="SR (output)").grid(row=0, column=0, sticky="w")
        self.sr_canvas = tk.Canvas(right, bg="#FFFFFF", highlightthickness=1, highlightbackground="#CCCCCC")
        self.sr_canvas.grid(row=1, column=0, sticky="nsew")
        self.sr_meta = ttk.Label(right, text="–")
        self.sr_meta.grid(row=2, column=0, sticky="w", pady=(4,0))

        body.add(left, weight=1)
        body.add(right, weight=1)

        # Keep references to PhotoImage to prevent GC
        self._lr_photo = None
        self._sr_photo = None

        # Light theme via ttk styles
        style = ttk.Style(self)
        try:
            style.theme_use("default")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#FFFFFF", foreground="#000000")
        style.configure("TButton", padding=6)
        style.configure("Horizontal.TProgressbar", troughcolor="#EEEEEE", background="#3B82F6")

        self.bind("<Configure>", self._on_resize)

    # ---------- Actions ----------
    def load_model(self):
        path = filedialog.askopenfilename(title="Select ONNX model",
                                          filetypes=[("ONNX Model", "*.onnx"), ("All Files", "*.*")])
        if not path:
            return
        try:
            # CPUExecutionProvider = most stable everywhere
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.io_names = get_io_names(self.sess)
            self.onnx_path = path
            self.status_var.set(f"Loaded model: {Path(path).name} • IO: {self.io_names[0]} → {self.io_names[1]}")
            self._refresh_buttons()
        except Exception as e:
            self.sess = None
            self.io_names = None
            messagebox.showerror("ONNX Runtime error", str(e))
            self.status_var.set("Failed to load model.")

    def load_image(self):
        path = filedialog.askopenfilename(title="Select LR image",
                                          filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        self.lr_bgr = img
        self._draw_lr()
        self.status_var.set(f"Loaded LR image: {Path(path).name} • {img.shape[1]}×{img.shape[0]}")
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.run_btn.config(state=("normal" if (self.sess is not None and self.lr_bgr is not None) else "disabled"))
        self.save_btn.config(state=("normal" if (self.sr_bgr is not None) else "disabled"))

    def run_inference(self):
        if self.sess is None or self.lr_bgr is None:
            return
        # Start progress
        self.pbar.start(10)
        self.run_btn.config(state="disabled")
        self.status_var.set("Running inference...")

        # Launch worker thread
        t = threading.Thread(target=self._infer_worker, daemon=True)
        t.start()

    def _infer_worker(self):
        try:
            t0 = time.time()
            window_size = int(self.window_size_var.get())
            normalize = self.normalize_var.get()

            # 1) pad LR
            lr_bgr_pad, ph, pw = pad_to_multiple(self.lr_bgr, window_size)

            # 2) BGR->RGB + normalize
            rgb = cv2.cvtColor(lr_bgr_pad, cv2.COLOR_BGR2RGB).astype(np.float32)
            if normalize == "0to1":
                rgb /= 255.0
            else:  # "-1to1"
                rgb = (rgb / 127.5) - 1.0

            inp = np.transpose(rgb, (2, 0, 1))[None, ...]  # 1x3xhxw
            in_name, out_name = self.io_names

            # 3) forward
            out = self.sess.run([out_name], {in_name: inp})[0]
            if out.ndim == 4:
                out = out[0]  # CHW
            sr = np.transpose(out, (1, 2, 0))  # HxWx3 (RGB)

            # 4) infer scale & crop padding
            scale = int(round(sr.shape[0] / lr_bgr_pad.shape[0]))
            if ph or pw:
                H, W = sr.shape[:2]
                sr = sr[: H - ph * scale, : W - pw * scale, :]

            # 5) denormalize + clamp + RGB->BGR
            if normalize == "0to1":
                sr = (np.clip(sr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                sr = ((np.clip(sr, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
            sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

            t1 = time.time()
            elapsed = t1 - t0

            # Push results to UI thread
            self.after(0, self._on_infer_done, sr_bgr, elapsed, scale)
        except Exception as e:
            self.after(0, self._on_infer_fail, str(e))

    def _on_infer_done(self, sr_bgr: np.ndarray, elapsed: float, scale: int):
        self.pbar.stop()
        self.sr_bgr = sr_bgr
        self._draw_sr()
        self.status_var.set(f"Inference OK • ×{scale} • {elapsed:.2f}s")
        self._refresh_buttons()
        self.run_btn.config(state="normal")

    def _on_infer_fail(self, err: str):
        self.pbar.stop()
        messagebox.showerror("Inference failed", err)
        self.status_var.set("Inference failed.")
        self.run_btn.config(state="normal")

    def save_sr(self):
        if self.sr_bgr is None:
            return
        path = filedialog.asksaveasfilename(title="Save SR image",
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"),
                                                       ("JPEG", "*.jpg;*.jpeg"),
                                                       ("BMP", "*.bmp")],
                                            initialfile="sr_x4.png")
        if not path:
            return
        ok = cv2.imwrite(path, self.sr_bgr)
        if ok:
            self.status_var.set(f"Saved: {path}")
        else:
            messagebox.showerror("Error", "Failed to save image.")

    # ---------- Drawing ----------
    def _draw_lr(self):
        self._draw_to_canvas(self.lr_bgr, self.lr_canvas, is_lr=True)

    def _draw_sr(self):
        self._draw_to_canvas(self.sr_bgr, self.sr_canvas, is_lr=False)

    def _draw_to_canvas(self, img_bgr: np.ndarray | None, canvas: tk.Canvas, is_lr: bool):
        if img_bgr is None:
            canvas.delete("all")
            return
        # Fit preview to canvas size (minus small padding)
        cw = max(100, canvas.winfo_width() - 8)
        ch = max(100, canvas.winfo_height() - 8)
        pil = bgr_to_pil(img_bgr)
        pil = fit_image_for_preview(pil, cw, ch)
        photo = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        # store ref
        if is_lr:
            self._lr_photo = photo
            self.lr_meta.config(text=f"{img_bgr.shape[1]} × {img_bgr.shape[0]}")
        else:
            self._sr_photo = photo
            self.sr_meta.config(text=f"{img_bgr.shape[1]} × {img_bgr.shape[0]}")

    def _on_resize(self, _evt):
        # redraw previews to fit new size
        if self.lr_bgr is not None:
            self._draw_lr()
        if self.sr_bgr is not None:
            self._draw_sr()


if __name__ == "__main__":
    app = SRApp()
    app.mainloop()

# Super-Resolution GUI Applications (Streamlit & Tkinter)

This repository provides two graphical interfaces for running Super-Resolution (SR) models exported to ONNX format:

- **Streamlit Web App** – A modern browser-based interface for SR inference.
- **Tkinter Desktop App** – A lightweight desktop application with native controls.

Both applications allow you to load ONNX-based SR models (e.g., SwinIR, MicroSR), upload low-resolution images, run inference, and display results side by side.

---

## 📦 Prerequisites

- Python 3.11 (recommended)
- ONNX Runtime
- OpenCV
- NumPy
- Pillow (for Tkinter GUI)
- Streamlit (for web GUI)

### 🔧 GPU Support (Optional)

If you want to enable GPU acceleration, install the following additional dependencies:

- `nvidia-cublas-cu12==12.9.1.4`
- `nvidia-cudnn-cu12==9.13.0.50`
- `onnxruntime-directml==1.22.0`
- `onnxruntime-gpu==1.22.0`

**Steps:**

```bash
# Install GPU-enabled ONNX Runtime
pip install onnxruntime-gpu

# Install cuDNN package
py -m pip install nvidia-cudnn-cu12==9.13.0.50
```

Download and install **CUDA 12.8**:  
[CUDA 12.8 Download](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

Reference cuDNN support matrix:  
[cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)

**Virtual Environment Config:**  
In your `pyvenv.cfg` file, set:

```
include-system-site-packages = true
```

**Set PATH variables (Windows):**

```bash
(sr_gui_env) C:\sr_gui\source> set PATH=C:\sr_gui\sr_gui_env\Lib\site-packages\nvidia\cudnn\bin;%PATH%
(sr_gui_env) C:\sr_gui\source> set PATH=C:\sr_gui\sr_gui_env\Lib\site-packages\nvidia\cublas\bin;%PATH%
```

---

## 🚀 Installation

1. **Create and activate a virtual environment**

   ```bash
   py -3.11 -m venv sr_gui_env
   sr_gui_env\Scripts\activate        # Windows
   source sr_gui_env/bin/activate       # Linux/Mac
   ```

2. **Install dependencies**

   ```bash
   cd source
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### Streamlit Web Application

To launch the Streamlit-based web GUI:

```bash
streamlit run app.py
```

After the app launches, open the provided URL (e.g., [http://localhost:8501](http://localhost:8501)) in your browser.

**Features:**
- Upload a pre-trained ONNX model (e.g., SwinIR, MicroSR)
- Upload a low-resolution image
- Adjust parameters (e.g., window size, normalization)
- Run inference
- View LR and SR images side by side

### Tkinter Desktop Application

To run the native desktop app using Tkinter:

```bash
python sr_tk_gui.py
```

**Features:**
- Load a pre-trained ONNX model
- Load a low-resolution input image
- Set window size and normalization flag
- Run inference with a progress bar
- Display and save the super-resolved output

---

## 📝 Notes

- Both GUIs are model-agnostic and support any ONNX-based SR model with a standard NCHW input/output format.
- The default runtime uses `CPUExecutionProvider` for broad compatibility.  
- GPU support (CUDAExecutionProvider or DirectML) can be enabled if properly configured with CUDA/cuDNN.
- For very large images, patch-based or tiled inference is recommended to prevent memory issues (not included in this demo).
- Inference performance depends on your hardware and model size.

---

## ⚠️ Troubleshooting

- **ONNX Runtime DLL Errors (Windows)**  
  If you encounter errors like `ImportError: DLL load failed`, ensure the following:
  - You are using the correct Python version (e.g., Python 3.11).
  - The virtual environment is activated before installing and running.
  - If GPU support is required, make sure CUDA and cuDNN are properly installed and match your ONNX Runtime version.

- **Streamlit fails to launch**  
  Ensure all required packages are installed, and check that no other process is using port `8501`.

- **Black window or UI glitches in Tkinter**  
  Ensure that the correct versions of Pillow and Tkinter are installed and compatible with your OS and Python version.

---

For questions, bug reports, or contributions, please open an issue or submit a pull request.

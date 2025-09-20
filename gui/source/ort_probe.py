import onnxruntime as ort, sys, os, platform

path = r"swinir_s_x4_lightweight.onnx"  # <-- set this

print("Python   :", sys.executable)
print("Platform :", platform.platform())
print("ORT file :", ort.__file__)
print("ORT ver  :", ort.__version__)
print("Avail EPs:", ort.get_available_providers())

print("\n--- Trying CUDA only (no CPU fallback) ---")
try:
    s = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    print("Session EPs:", s.get_providers())
    print("Provider opts:", s.get_provider_options())
    print("CUDA-only session created OK ✅")
except Exception as e:
    print("CUDA-only session FAILED ❌")
    print(e)

print("\n--- Trying DirectML only ---")
try:
    s = ort.InferenceSession(path, providers=["DmlExecutionProvider"])
    print("Session EPs:", s.get_providers())
    print("DirectML-only session created OK ✅")
except Exception as e:
    print("DirectML-only session FAILED ❌")
    print(e)

print("\n--- CPU only ---")
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("Session EPs:", s.get_providers())
print("CPU-only session OK ✅")

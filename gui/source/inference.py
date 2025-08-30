import cv2, numpy as np, onnxruntime as ort

onnx_path = "swinir_s_x4_lightweight.onnx"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def pad_to_multiple(img, m=8):
    h,w = img.shape[:2]
    ph = (m - h % m) % m; pw = (m - w % m) % m
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return img, ph, pw

def swinir_x4_ort(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb, ph, pw = pad_to_multiple(rgb, 8)
    inp = np.transpose(rgb,(2,0,1))[None]
    out = sess.run(None, {"lr": inp})[0][0]              # CHW, [0,1]
    sr = np.transpose(out,(1,2,0))
    if ph or pw:
        H,W = sr.shape[:2]; sr = sr[:H - ph*4, :W - pw*4]
    sr = np.clip(sr,0,1)
    return cv2.cvtColor((sr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Example
lr = cv2.imread("0006_lr.png")  # low-res image
sr = swinir_x4_ort(lr)
cv2.imwrite("sr_x4.png", sr)
print("Saved sr_x4.png")

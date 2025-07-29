import torch
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import transforms

# Assuming the necessary imports and helper functions (_prepare_teacher_input, _get_teacher_output, _patch_based_inference, load_teacher, load_student) are available from previous cells.

# Helper function to prepare input for teacher model
def _prepare_teacher_input(lr_tensor, window_size=16):
    """Prepare input for teacher model following your preprocessing pipeline"""
    # Ensure tensor is on correct device
    device = lr_tensor.device
    if lr_tensor.dim() == 3:
        lr_tensor = lr_tensor.unsqueeze(0)

    # Pad to window size multiples
    _, _, h_old, w_old = lr_tensor.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old

    if h_pad > 0 or w_pad > 0:
        lr_padded = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        lr_padded = torch.cat([lr_padded, torch.flip(lr_padded, [3])], 3)[:, :, :, :w_old + w_pad]
    else:
        lr_padded = lr_tensor

    return lr_padded, (h_old, w_old)

# Helper function to get teacher output
def _get_teacher_output(lr_tensor, teacher_model):
    """Get teacher model output by applying necessary padding and forwarding through the model"""
    with torch.no_grad():
        # Ensure input is on correct device before padding
        device = lr_tensor.device
        lr_tensor = lr_tensor.to(device)

        lr_padded, (h_old, w_old) = _prepare_teacher_input(lr_tensor)

        # Pass the padded tensor directly to the teacher model's forward method
        sr_padded = teacher_model(lr_padded)

        # Crop to original size * scale_factor (assuming scale factor is 4)
        sr_output = sr_padded[..., :h_old * 4, :w_old * 4]
        return sr_output


# Helper function for patch-based inference
def _patch_based_inference(model, lr_img, patch_size=64, overlap=16):
    """Perform patch-based inference with overlapping"""
    device = lr_img.device  # Get device from input tensor
    lr_img = lr_img.to(device)  # Ensure on correct device
    _, h, w = lr_img.shape
    sr_img = torch.zeros((3, h * 4, w * 4), device=device)
    count_map = torch.zeros((h * 4, w * 4), device=device)

    stride = patch_size - overlap

    # Ensure we don't go out of bounds
    y_positions = list(range(0, max(1, h - patch_size + 1), stride))
    if y_positions[-1] != h - patch_size and h >= patch_size:
        y_positions.append(h - patch_size)

    x_positions = list(range(0, max(1, w - patch_size + 1), stride))
    if x_positions[-1] != w - patch_size and w >= patch_size:
        x_positions.append(w - patch_size)


    for y in y_positions:
        for x in x_positions:
            # Extract patch
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch = lr_img[:, y:y_end, x:x_end]

            # Pad if necessary
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                pad_h = patch_size - patch.shape[1]
                pad_w = patch_size - patch.shape[2]
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')

            # Super-resolve patch
            sr_patch = model(patch.unsqueeze(0)).squeeze(0)

            # Crop if we padded
            sr_patch = sr_patch[:, :4*(y_end-y), :4*(x_end-x)]

            # Place in output image
            y_sr, x_sr = y * 4, x * 4
            y_sr_end, x_sr_end = y_sr + sr_patch.shape[1], x_sr + sr_patch.shape[2]

            sr_img[:, y_sr:y_sr_end, x_sr:x_sr_end] += sr_patch
            count_map[y_sr:y_sr_end, x_sr:x_sr_end] += 1

    # Average overlapping regions
    count_map = torch.clamp(count_map, min=1)
    sr_img = sr_img / count_map.unsqueeze(0)

    return sr_img


# Helper function to load the teacher (MicroSR) model
# Removed WrappedTeacher class
def load_teacher(device):
    """Load the teacher (MicroSR) model"""
    model_dir = "/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4"
    teacher = MicroSR(
        upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['params'], strict=True)
    # Return the raw teacher model
    return teacher.to(device)


# Helper function to load the student (SwinIR) model from checkpoint
def load_student(device):
    """Load the student (SwinIR) model from checkpoint"""
    student = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64, # Note: img_size here is for the model definition, not patch size
        window_size=8,
        img_range=1.0,
        depths=[2, 2, 2, 2],            # fewer transformer layers
        embed_dim=60,                   # smaller embedding size
        num_heads=[2, 2, 2, 2],         # fewer attention heads
        mlp_ratio=1.5,                  # reduced MLP size
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    # Use the 'best_student_model.pth' saved during training
    student_checkpoint_path = 'best_student_model.pth'
    if not os.path.exists(student_checkpoint_path):
        raise FileNotFoundError(f"Student model checkpoint not found at {student_checkpoint_path}")

    checkpoint_student = torch.load(student_checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint_student['model_state_dict'], strict=True)
    return student.to(device)


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running inference on device: {device}')

# Load the dummy low-resolution image
img_lr_path = "/content/div2k_val_0801_lr.png"
if not os.path.exists(img_lr_path):
    raise FileNotFoundError(f"Low-resolution image not found at {img_lr_path}")

# Read the image using OpenCV and convert to RGB (Matplotlib expects RGB)
img_lr_np = cv2.imread(img_lr_path)
img_lr_np = img_lr_np[200:300, 200:300, :]
img_lr_np = cv2.cvtColor(img_lr_np, cv2.COLOR_BGR2RGB)

# Convert to tensor and normalize for model input
# Models expect [0, 1] range
img_lr_tensor = transforms.ToTensor()(img_lr_np).unsqueeze(0).to(device)


# Load Teacher model
print("Loading teacher model...")
teacher_model = load_teacher(device)
teacher_model.eval()

# Load Student model
print("Loading student model...")
student_model = load_student(device)
student_model.eval()

# Perform inference with Teacher model
print("Performing inference with teacher model...")
with torch.no_grad():
    # Use the modified _get_teacher_output function
    img_sr_teacher_tensor = _get_teacher_output(img_lr_tensor, teacher_model)
    img_sr_teacher_tensor = torch.clamp(img_sr_teacher_tensor, 0, 1) # Clamp to valid range

# Perform inference with Student model using patch-based approach
print("Performing inference with student model...")
with torch.no_grad():
    # Remove batch dimension for patch-based inference function
    img_sr_student_tensor = _patch_based_inference(student_model, img_lr_tensor.squeeze(0))
    img_sr_student_tensor = torch.clamp(img_sr_student_tensor, 0, 1) # Clamp to valid range


# Convert SR tensors back to NumPy for display
img_sr_teacher_np = img_sr_teacher_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
img_sr_student_np = img_sr_student_tensor.permute(1, 2, 0).cpu().numpy()

# Display the images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_lr_np)
plt.title("Low Resolution (LR)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_sr_teacher_np)
plt.title("Teacher (MicroSR) Output")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_sr_student_np)
plt.title("Student (SwinIR) Output")
plt.axis('off')

plt.show()
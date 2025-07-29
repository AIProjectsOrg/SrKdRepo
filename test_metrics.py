import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from models.team07_MicroSR.model import MicroSR
from basicsr.archs.swinir_arch import SwinIR
from models.team07_MicroSR.io import forward # Assuming forward is needed for teacher

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
    """Get teacher model output"""
    with torch.no_grad():
        # Ensure input is on correct device
        lr_tensor = lr_tensor.to(teacher_model.device)
        lr_padded, (h_old, w_old) = _prepare_teacher_input(lr_tensor)
        # Assuming the 'forward' function is compatible with the teacher model's signature
        sr_padded = forward(lr_padded, teacher_model, tile=None)
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


def evaluate_model(model, dataset, device, model_name):
    """Evaluate model performance (PSNR, LPIPS) on the dataset"""
    model.eval()
    total_psnr = 0
    total_lpips = 0
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f'Evaluating {model_name}'):
            hr_img = dataset[i].to(device)  # Move to device

            # Downsample to create LR
            lr_img = torch.nn.functional.interpolate(
                hr_img.unsqueeze(0),
                scale_factor=0.25,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)

            # Get SR image using patch-based inference
            sr_img = _patch_based_inference(model, lr_img)

            # Calculate metrics
            sr_clamped = torch.clamp(sr_img, 0, 1)
            hr_clamped = torch.clamp(hr_img, 0, 1) # Compare against original HR

            sr_np = sr_clamped.cpu().numpy().transpose(1, 2, 0)
            hr_np = hr_clamped.cpu().numpy().transpose(1, 2, 0)

            img_psnr = psnr(hr_np, sr_np, data_range=1.0)
            img_lpips = lpips_loss(sr_clamped.unsqueeze(0), hr_clamped.unsqueeze(0)).item()

            total_psnr += img_psnr
            total_lpips += img_lpips

    avg_psnr = total_psnr / len(dataset)
    avg_lpips = total_lpips / len(dataset)

    print(f"\n==== {model_name} Performance ====")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    return avg_psnr, avg_lpips


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running evaluation on device: {device}')

# Load validation dataset
print("Creating validation dataset...")
# Re-using the dataset class defined previously
val_dataset = SuperResolutionDataset(
    "/content/DIV2K_HR/DIV2K_valid_HR/DIV2K_valid_HR",
    patch_size=256, # Patch size doesn't matter for validation dataset loading
    scale_factor=4,
    is_training=False
)


# Load Teacher model (MicroSR)
print("Loading teacher model...")
teacher_model_dir = "/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4"
teacher_model = MicroSR(
    upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
    squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
    depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
)
checkpoint_teacher = torch.load(teacher_model_dir, map_location=device, weights_only=False)
teacher_model.load_state_dict(checkpoint_teacher['params'], strict=True)
teacher_model.to(device)
teacher_model.eval()


# Load Student model (SwinIR) from checkpoint
print("Loading student model from checkpoint...")
student_model = SwinIR(
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
student_model.load_state_dict(checkpoint_student['model_state_dict'], strict=True)
student_model.to(device)
student_model.eval()


# Evaluate models
print("\nStarting evaluation...")
teacher_psnr, teacher_lpips = evaluate_model(teacher_model, val_dataset, device, "Teacher (MicroSR)")
student_psnr, student_lpips = evaluate_model(student_model, val_dataset, device, "Student (SwinIR)")

print("\n==== Comparison ====")
print(f"Teacher (MicroSR) - PSNR: {teacher_psnr:.2f}, LPIPS: {teacher_lpips:.4f}")
print(f"Student (SwinIR) - PSNR: {student_psnr:.2f}, LPIPS: {student_lpips:.4f}")
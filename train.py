import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import logging
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
from utils import utils_image as util

# Import your teacher model components
from models.team07_MicroSR.model import MicroSR
from models.team07_MicroSR.io import forward

# Import BasicSR components
try:
    from basicsr.archs.swinir_arch import SwinIR
except ImportError:
    print("Warning: BasicSR not found. Please install BasicSR for SwinIR model.")

class SuperResolutionDataset(Dataset):
    """Dataset for super resolution training"""

    def __init__(self, hr_folder, patch_size=256, scale_factor=4, is_training=True):
        self.hr_folder = hr_folder
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_training = is_training
        self.lr_size = patch_size // scale_factor

        # Get all image paths
        self.image_paths = glob.glob(os.path.join(hr_folder, "*.png")) + \
                          glob.glob(os.path.join(hr_folder, "*.jpg")) + \
                          glob.glob(os.path.join(hr_folder, "*.jpeg"))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {hr_folder}")

        print(f"Found {len(self.image_paths)} images in {hr_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load HR image
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert('RGB')
        hr_img = np.array(hr_img)

        if self.is_training:
            # Random crop for training
            h, w = hr_img.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                # Resize if image is smaller than patch size
                scale = max(self.patch_size / h, self.patch_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                hr_img = np.array(Image.fromarray(hr_img).resize((new_w, new_h), Image.BICUBIC))
                h, w = hr_img.shape[:2]

            # Random crop
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]

            # Convert to tensor and normalize
            hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0

            # Downsample to create LR using CPU operations, then move to device later
            lr_tensor = torch.nn.functional.interpolate(
                hr_tensor.unsqueeze(0),
                size=(self.lr_size, self.lr_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)

            return lr_tensor, hr_tensor
        else:
            # For validation, return full image
            hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
            return hr_tensor

class SuperResolutionTrainer:
    """Main training class for super resolution with knowledge distillation"""

    def __init__(self, teacher_model, student_model_class=SwinIR, device='cuda'):
        self.device = device
        self.teacher_model = teacher_model.to(device)  # Ensure teacher is on correct device
        self.teacher_model.eval()

        # Initialize student model (configurable)
        self.student_model = self._create_student_model(student_model_class)
        self.student_model.to(device)

        # Loss functions - ensure LPIPS is on correct device
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Training settings
        self.loss_weights = {'mse': 1.0, 'lpips': 0.1}
        self.window_size = 16

    def _create_student_model(self, model_class):
        """Create a lightweight student model"""
        if model_class == SwinIR:
            return SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[2, 2, 2, 2],            # fewer transformer layers
                embed_dim=60,                   # smaller embedding size
                num_heads=[2, 2, 2, 2],         # fewer attention heads
                mlp_ratio=1.5,                  # reduced MLP size
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )
        else:
            raise NotImplementedError(f"Student model {model_class} not implemented")


    def _prepare_teacher_input(self, lr_tensor):
        """Prepare input for teacher model following your preprocessing pipeline"""
        # Ensure tensor is on correct device
        lr_tensor = lr_tensor.to(self.device)

        # Convert to 4D tensor if needed
        if lr_tensor.dim() == 3:
            lr_tensor = lr_tensor.unsqueeze(0)

        # Pad to window size multiples
        _, _, h_old, w_old = lr_tensor.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

        if h_pad > 0 or w_pad > 0:
            lr_padded = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            lr_padded = torch.cat([lr_padded, torch.flip(lr_padded, [3])], 3)[:, :, :, :w_old + w_pad]
        else:
            lr_padded = lr_tensor

        return lr_padded, (h_old, w_old)

    def _get_teacher_output(self, lr_tensor):
        """Get teacher model output"""
        with torch.no_grad():
            # Ensure input is on correct device
            lr_tensor = lr_tensor.to(self.device)
            lr_padded, (h_old, w_old) = self._prepare_teacher_input(lr_tensor)
            sr_padded = forward(lr_padded, self.teacher_model, tile=None)
            # Crop to original size * scale_factor
            sr_output = sr_padded[..., :h_old * 4, :w_old * 4]
            return sr_output

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0
        total_mse = 0
        total_lpips = 0
        total_psnr = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lr_batch, hr_batch) in enumerate(pbar):
            # Move data to device FIRST
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)

            # Get teacher outputs
            teacher_outputs = []
            for i in range(lr_batch.size(0)):
                teacher_out = self._get_teacher_output(lr_batch[i:i+1])
                teacher_outputs.append(teacher_out)
            teacher_batch = torch.cat(teacher_outputs, dim=0)

            # Get student output
            student_output = self.student_model(lr_batch)

            # Ensure both outputs are on same device and same size
            if teacher_batch.shape != student_output.shape:
                print(f"Shape mismatch: teacher {teacher_batch.shape}, student {student_output.shape}")
                # Resize if needed
                teacher_batch = torch.nn.functional.interpolate(
                    teacher_batch,
                    size=student_output.shape[-2:],
                    mode='bicubic',
                    align_corners=False
                )

            # Calculate losses
            mse_loss = self.mse_loss(student_output, teacher_batch)

            # Ensure tensors are in correct range for LPIPS (0-1 or -1 to 1)
            student_clamped = torch.clamp(student_output, 0, 1)
            teacher_clamped = torch.clamp(teacher_batch, 0, 1)
            lpips_loss = self.lpips_loss(student_clamped, teacher_clamped).mean()

            total_loss_batch = (self.loss_weights['mse'] * mse_loss +
                              self.loss_weights['lpips'] * lpips_loss)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            # Calculate PSNR
            with torch.no_grad():
                batch_psnr = 0
                for i in range(student_output.size(0)):
                    student_np = torch.clamp(student_output[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    teacher_np = torch.clamp(teacher_batch[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    batch_psnr += psnr(teacher_np, student_np, data_range=1.0)
                batch_psnr /= student_output.size(0)

            # Update running averages
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_lpips += lpips_loss.item()
            total_psnr += batch_psnr

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'LPIPS': f'{lpips_loss.item():.4f}',
                'PSNR': f'{batch_psnr:.2f}'
            })

        return {
            'loss': total_loss / len(train_loader),
            'mse': total_mse / len(train_loader),
            'lpips': total_lpips / len(train_loader),
            'psnr': total_psnr / len(train_loader)
        }

    def validate_epoch(self, val_dataset):
        """Validate on full images with patch-based inference"""
        self.student_model.eval()
        total_psnr = 0
        total_lpips = 0

        with torch.no_grad():
            for i in tqdm(range(min(10, len(val_dataset))), desc='Validation'):  # Limit validation for speed
                hr_img = val_dataset[i].to(self.device)  # Move to device

                # Downsample to create LR
                lr_img = torch.nn.functional.interpolate(
                    hr_img.unsqueeze(0),
                    scale_factor=0.25,
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)

                # Patch-based inference
                sr_img = self._patch_based_inference(lr_img)

                # Get teacher output for comparison
                teacher_out = self._get_teacher_output(lr_img.unsqueeze(0)).squeeze(0)

                # Ensure same size
                if teacher_out.shape != sr_img.shape:
                    teacher_out = torch.nn.functional.interpolate(
                        teacher_out.unsqueeze(0),
                        size=sr_img.shape[-2:],
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)

                # Calculate metrics
                sr_clamped = torch.clamp(sr_img, 0, 1)
                teacher_clamped = torch.clamp(teacher_out, 0, 1)

                sr_np = sr_clamped.cpu().numpy().transpose(1, 2, 0)
                teacher_np = teacher_clamped.cpu().numpy().transpose(1, 2, 0)

                img_psnr = psnr(teacher_np, sr_np, data_range=1.0)
                img_lpips = self.lpips_loss(sr_clamped.unsqueeze(0), teacher_clamped.unsqueeze(0)).item()

                total_psnr += img_psnr
                total_lpips += img_lpips

        return {
            'psnr': total_psnr / min(10, len(val_dataset)),
            'lpips': total_lpips / min(10, len(val_dataset))
        }

    def _patch_based_inference(self, lr_img, patch_size=64, overlap=16):  # Reduced overlap for speed
        """Perform patch-based inference with overlapping"""
        lr_img = lr_img.to(self.device)  # Ensure on correct device
        _, h, w = lr_img.shape
        sr_img = torch.zeros((3, h * 4, w * 4), device=self.device)
        count_map = torch.zeros((h * 4, w * 4), device=self.device)

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
                sr_patch = self.student_model(patch.unsqueeze(0)).squeeze(0)

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

def main():
    """Main training function"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')

    # Load teacher model (using your existing code)
    print("Loading teacher model...")
    model_dir = "/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4"
    teacher_model = MicroSR(
        upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )

    # Load state dict and move to device
    checkpoint = torch.load(model_dir, map_location=device)
    teacher_model.load_state_dict(checkpoint['params'], strict=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Create datasets
    print("Creating datasets...")
    train_dataset = SuperResolutionDataset(
        "/content/DIV2K_HR/DIV2K_train_HR/DIV2K_train_HR",
        patch_size=256,
        scale_factor=4,
        is_training=True
    )

    val_dataset = SuperResolutionDataset(
        "/content/DIV2K_HR/DIV2K_valid_HR/DIV2K_valid_HR",
        patch_size=256,
        scale_factor=4,
        is_training=False
    )

    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Reduced batch size to avoid memory issues
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize trainer
    trainer = SuperResolutionTrainer(teacher_model, SwinIR, device)

    # Training loop
    num_epochs = 100
    best_psnr = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        try:
            # Train
            train_metrics = trainer.train_epoch(train_loader, epoch+1)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"LPIPS: {train_metrics['lpips']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}")

            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                val_metrics = trainer.validate_epoch(val_dataset)
                print(f"Val - PSNR: {val_metrics['psnr']:.2f}, "
                      f"LPIPS: {val_metrics['lpips']:.4f}")

                # Save best model
                if val_metrics['psnr'] > best_psnr:
                    best_psnr = val_metrics['psnr']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.student_model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'psnr': best_psnr,
                    }, 'best_student_model.pth')
                    print(f"New best model saved with PSNR: {best_psnr:.2f}")

            # Update learning rate
            trainer.scheduler.step()

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.student_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                }, f'checkpoint_epoch_{epoch+1}.pth')

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory at epoch {epoch+1}. Try reducing batch size.")
                torch.cuda.empty_cache()
                break
            else:
                raise e

if __name__ == "__main__":
    main()
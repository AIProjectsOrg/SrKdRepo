"""
Inference utilities for super resolution models.
Handles preprocessing, postprocessing, and patch-based inference.
"""
import torch
import numpy as np


class InferenceProcessor:
    """Handles model inference with preprocessing and postprocessing"""
    
    def __init__(self, window_size=16, patch_size=64, overlap=16):
        self.window_size = window_size
        self.patch_size = patch_size
        self.overlap = overlap
    
    def prepare_teacher_input(self, lr_tensor, window_size=None):
        """
        Prepare input for teacher model following preprocessing pipeline
        
        Args:
            lr_tensor: Low resolution input tensor
            window_size: Window size for padding (default: self.window_size)
        """
        if window_size is None:
            window_size = self.window_size
            
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
    
    def get_teacher_output(self, lr_tensor, teacher_model, use_forward_func=True):
        """
        Get teacher model output with proper preprocessing
        
        Args:
            lr_tensor: Low resolution input
            teacher_model: Teacher model
            use_forward_func: Whether to use the forward function or direct model call
        """
        with torch.no_grad():
            device = lr_tensor.device
            lr_tensor = lr_tensor.to(device)
            
            lr_padded, (h_old, w_old) = self.prepare_teacher_input(lr_tensor)
            
            if use_forward_func:
                # Use the forward function from team07_MicroSR.io
                from models.team07_MicroSR.io import forward
                sr_padded = forward(lr_padded, teacher_model, tile=None)
            else:
                # Direct model call
                sr_padded = teacher_model(lr_padded)
            
            # Crop to original size * scale_factor (assuming scale factor is 4)
            sr_output = sr_padded[..., :h_old * 4, :w_old * 4]
            return sr_output
    
    def patch_based_inference(self, model, lr_img, patch_size=None, overlap=None):
        """
        Perform patch-based inference with overlapping
        
        Args:
            model: The model to use for inference
            lr_img: Low resolution image tensor (C, H, W)
            patch_size: Size of patches (default: self.patch_size)
            overlap: Overlap between patches (default: self.overlap)
        """
        if patch_size is None:
            patch_size = self.patch_size
        if overlap is None:
            overlap = self.overlap
            
        device = lr_img.device
        lr_img = lr_img.to(device)
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
    
    def process_image_for_display(self, tensor):
        """Convert tensor to numpy array for display"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.permute(1, 2, 0).cpu().numpy()

"""
Model loading utilities for teacher and student models.
Consolidates all model loading logic in one place.
"""
import torch
import os
from .team07_MicroSR.model import MicroSR
from .team07_MicroSR.io import forward
try:
    from basicsr.archs.swinir_arch import SwinIR
except ImportError:
    print("Warning: BasicSR not found. Please install BasicSR for SwinIR model.")
    SwinIR = None


class WrappedTeacher(torch.nn.Module):
    """Wrap teacher model to match forward(x) API"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return forward(x, self.model, tile=None)


class ModelLoader:
    """Centralized model loading class"""
    
    @staticmethod
    def load_teacher(device, model_path="/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4", wrapped=False):
        """
        Load the teacher (MicroSR) model
        
        Args:
            device: torch device
            model_path: path to the teacher model checkpoint
            wrapped: whether to wrap the model for standardized forward API
        """
        teacher = MicroSR(
            upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
            squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        teacher.load_state_dict(checkpoint['params'], strict=True)
        teacher = teacher.to(device)
        
        if wrapped:
            return WrappedTeacher(teacher)
        return teacher
    
    @staticmethod
    def load_student(device, checkpoint_path='best_student_model.pth'):
        """
        Load the student (SwinIR) model from checkpoint
        
        Args:
            device: torch device
            checkpoint_path: path to the student model checkpoint
        """
        if SwinIR is None:
            raise ImportError("SwinIR not available. Please install BasicSR.")
            
        student = SwinIR(
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
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Student model checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return student.to(device)
    
    @staticmethod
    def create_student_model():
        """Create a new student model for training"""
        if SwinIR is None:
            raise ImportError("SwinIR not available. Please install BasicSR.")
            
        return SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[2, 2, 2, 2],
            embed_dim=60,
            num_heads=[2, 2, 2, 2],
            mlp_ratio=1.5,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

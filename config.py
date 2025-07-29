"""
Configuration settings for the super resolution project.
Centralizes all configuration in one place.
"""
import os


class Config:
    """Configuration class with all project settings"""
    
    # Model paths
    TEACHER_MODEL_PATH = "/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4"
    STUDENT_CHECKPOINT_PATH = "best_student_model.pth"
    
    # Dataset paths
    TRAIN_HR_FOLDER = "/content/DIV2K_HR/DIV2K_train_HR/DIV2K_train_HR"
    VAL_HR_FOLDER = "/content/DIV2K_HR/DIV2K_valid_HR/DIV2K_valid_HR"
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATCH_SIZE = 256
    SCALE_FACTOR = 4
    
    # Model parameters
    TEACHER_WINDOW_SIZE = 16
    STUDENT_WINDOW_SIZE = 8
    
    # Inference parameters
    INFERENCE_PATCH_SIZE = 64
    INFERENCE_OVERLAP = 16
    
    # Loss weights
    MSE_WEIGHT = 1.0
    LPIPS_WEIGHT = 0.1
    
    # Benchmark parameters
    BENCHMARK_WARMUP = 10
    BENCHMARK_RUNS = 50
    BENCHMARK_INPUT_SIZE = 64
    
    # Device
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Validation settings
    VAL_INTERVAL = 5  # Validate every N epochs
    SAVE_BEST_ONLY = True
    
    @classmethod
    def update_paths(cls, **kwargs):
        """Update configuration paths"""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

"""
Modular evaluation script for testing model performance metrics.
"""
import torch

from config import Config
from models.model_loader import ModelLoader
from utils.dataset import SuperResolutionDataset
from utils.evaluation import ModelEvaluator
from utils.inference import InferenceProcessor


def main():
    """Main evaluation function"""
    # Setup
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running evaluation on device: {device}')
    
    # Initialize components
    evaluator = ModelEvaluator(device)
    inference_processor = InferenceProcessor(
        window_size=config.TEACHER_WINDOW_SIZE,
        patch_size=config.INFERENCE_PATCH_SIZE,
        overlap=config.INFERENCE_OVERLAP
    )
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = SuperResolutionDataset(
        config.VAL_HR_FOLDER,
        patch_size=config.PATCH_SIZE,
        scale_factor=config.SCALE_FACTOR,
        is_training=False
    )
    
    # Load models
    print("Loading models...")
    teacher_model = ModelLoader.load_teacher(device, config.TEACHER_MODEL_PATH)
    teacher_model.eval()
    
    student_model = ModelLoader.load_student(device, config.STUDENT_CHECKPOINT_PATH)
    student_model.eval()
    
    # Evaluate models
    print("\nStarting evaluation...")
    
    # Evaluate teacher
    teacher_psnr, teacher_lpips = evaluator.evaluate_model(
        teacher_model, 
        val_dataset, 
        "Teacher (MicroSR)",
        use_teacher_inference=True,
        inference_processor=inference_processor
    )
    
    # Evaluate student
    student_psnr, student_lpips = evaluator.evaluate_model(
        student_model, 
        val_dataset, 
        "Student (SwinIR)"
    )
    
    # Print comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Teacher (MicroSR) - PSNR: {teacher_psnr:.2f}, LPIPS: {teacher_lpips:.4f}")
    print(f"Student (SwinIR)  - PSNR: {student_psnr:.2f}, LPIPS: {student_lpips:.4f}")
    print(f"PSNR Difference:  {student_psnr - teacher_psnr:.2f}")
    print(f"LPIPS Difference: {student_lpips - teacher_lpips:.4f}")


if __name__ == "__main__":
    main()

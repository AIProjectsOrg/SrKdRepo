"""
Modular inference script for super resolution models.
"""
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os

from config import Config
from models.model_loader import ModelLoader
from utils.inference import InferenceProcessor


def load_test_image(image_path, crop_region=None):
    """
    Load and preprocess test image
    
    Args:
        image_path: Path to the test image
        crop_region: Optional tuple (top, bottom, left, right) for cropping
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")

    # Read image
    img_np = cv2.imread(image_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Apply crop if specified
    if crop_region is not None:
        top, bottom, left, right = crop_region
        img_np = img_np[top:bottom, left:right, :]
    
    # Convert to tensor and normalize
    img_tensor = transforms.ToTensor()(img_np).unsqueeze(0)
    
    return img_tensor, img_np


def run_inference(teacher_model, student_model, lr_tensor, inference_processor, device):
    """
    Run inference with both teacher and student models
    
    Args:
        teacher_model: Loaded teacher model
        student_model: Loaded student model
        lr_tensor: Low resolution input tensor
        inference_processor: InferenceProcessor instance
        device: Computing device
    """
    lr_tensor = lr_tensor.to(device)
    
    # Teacher inference
    print("Running teacher inference...")
    with torch.no_grad():
        teacher_output = inference_processor.get_teacher_output(
            lr_tensor, teacher_model, use_forward_func=True
        )
        teacher_output = torch.clamp(teacher_output, 0, 1)
    
    # Student inference (patch-based)
    print("Running student inference...")
    with torch.no_grad():
        student_output = inference_processor.patch_based_inference(
            student_model, lr_tensor.squeeze(0)
        )
        student_output = torch.clamp(student_output, 0, 1)
    
    return teacher_output, student_output


def display_results(lr_image, teacher_output, student_output, save_path=None):
    """
    Display inference results
    
    Args:
        lr_image: Original low resolution image (numpy array)
        teacher_output: Teacher model output tensor
        student_output: Student model output tensor
        save_path: Optional path to save the comparison image
    """
    # Convert tensors to numpy arrays
    teacher_np = teacher_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    student_np = student_output.permute(1, 2, 0).cpu().numpy()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(lr_image)
    axes[0].set_title("Low Resolution Input")
    axes[0].axis('off')
    
    axes[1].imshow(teacher_np)
    axes[1].set_title("Teacher (MicroSR) Output")
    axes[1].axis('off')
    
    axes[2].imshow(student_np)
    axes[2].set_title("Student (SwinIR) Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()


def main():
    """Main inference function"""
    # Setup
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running inference on device: {device}')
    
    # Initialize inference processor
    inference_processor = InferenceProcessor(
        window_size=config.TEACHER_WINDOW_SIZE,
        patch_size=config.INFERENCE_PATCH_SIZE,
        overlap=config.INFERENCE_OVERLAP
    )
    
    # Load models
    print("Loading models...")
    teacher_model = ModelLoader.load_teacher(device, config.TEACHER_MODEL_PATH)
    teacher_model.eval()
    
    student_model = ModelLoader.load_student(device, config.STUDENT_CHECKPOINT_PATH)
    student_model.eval()
    
    # Load test image
    test_image_path = "/content/div2k_val_0801_lr.png"
    crop_region = (200, 300, 200, 300)  # Optional cropping
    
    try:
        lr_tensor, lr_numpy = load_test_image(test_image_path, crop_region)
    except FileNotFoundError:
        print(f"Test image not found. Please update the path in the script.")
        return
    
    # Run inference
    teacher_output, student_output = run_inference(
        teacher_model, student_model, lr_tensor, inference_processor, device
    )
    
    # Display results
    display_results(
        lr_numpy, teacher_output, student_output,
        save_path="inference_comparison.png"
    )
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

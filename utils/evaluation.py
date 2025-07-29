"""
Evaluation utilities for model performance assessment.
"""
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm


class ModelEvaluator:
    """Handles model evaluation with various metrics"""
    
    def __init__(self, device):
        self.device = device
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
    
    def calculate_psnr(self, sr_img, hr_img):
        """Calculate PSNR between two images"""
        sr_np = sr_img.cpu().numpy()
        hr_np = hr_img.cpu().numpy()
        return psnr(hr_np, sr_np, data_range=1.0)
    
    def calculate_lpips(self, sr_img, hr_img):
        """Calculate LPIPS between two images"""
        # Ensure images are in the right format for LPIPS ([-1, 1] range)
        sr_lpips = sr_img * 2.0 - 1.0
        hr_lpips = hr_img * 2.0 - 1.0
        
        if sr_lpips.dim() == 3:
            sr_lpips = sr_lpips.unsqueeze(0)
        if hr_lpips.dim() == 3:
            hr_lpips = hr_lpips.unsqueeze(0)
            
        return self.lpips_loss(sr_lpips, hr_lpips).item()
    
    def evaluate_model(self, model, dataset, model_name="Model", use_teacher_inference=False, inference_processor=None):
        """
        Evaluate model performance on a dataset
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            model_name: Name for logging
            use_teacher_inference: Whether to use teacher-specific inference
            inference_processor: InferenceProcessor instance for teacher models
        """
        model.eval()
        total_psnr = 0
        total_lpips = 0

        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f'Evaluating {model_name}'):
                if hasattr(dataset, 'is_training') and not dataset.is_training:
                    # Validation dataset returns only HR
                    hr_img = dataset[i].to(self.device)
                    # Create LR by downsampling
                    lr_img = torch.nn.functional.interpolate(
                        hr_img.unsqueeze(0),
                        scale_factor=0.25,
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)
                else:
                    # Training dataset returns LR, HR
                    lr_img, hr_img = dataset[i]
                    lr_img = lr_img.to(self.device)
                    hr_img = hr_img.to(self.device)

                # Generate SR image
                if use_teacher_inference and inference_processor:
                    sr_img = inference_processor.get_teacher_output(lr_img.unsqueeze(0), model)
                    sr_img = sr_img.squeeze(0)
                else:
                    if lr_img.dim() == 3:
                        lr_img_input = lr_img.unsqueeze(0)
                    else:
                        lr_img_input = lr_img
                    sr_img = model(lr_img_input).squeeze(0)

                # Ensure images are in [0, 1] range
                sr_img = torch.clamp(sr_img, 0, 1)
                hr_img = torch.clamp(hr_img, 0, 1)

                # Resize to match if necessary
                if sr_img.shape != hr_img.shape:
                    sr_img = torch.nn.functional.interpolate(
                        sr_img.unsqueeze(0),
                        size=hr_img.shape[-2:],
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)

                # Calculate metrics
                psnr_val = self.calculate_psnr(sr_img, hr_img)
                lpips_val = self.calculate_lpips(sr_img, hr_img)

                total_psnr += psnr_val
                total_lpips += lpips_val

        avg_psnr = total_psnr / len(dataset)
        avg_lpips = total_lpips / len(dataset)

        print(f"\n==== {model_name} Performance ====")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        
        return avg_psnr, avg_lpips

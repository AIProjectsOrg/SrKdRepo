"""
Modular training script for super resolution with knowledge distillation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from tqdm import tqdm

from config import Config
from models.model_loader import ModelLoader
from utils.dataset import SuperResolutionDataset, create_dataloader
from utils.inference import InferenceProcessor
from utils.evaluation import ModelEvaluator

try:
    import lpips
except ImportError:
    print("Warning: lpips not found. Please install lpips for perceptual loss.")
    lpips = None


class SuperResolutionTrainer:
    """Main training class for super resolution with knowledge distillation"""

    def __init__(self, config=None):
        if config is None:
            config = Config
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Training on device: {self.device}')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.inference_processor = InferenceProcessor(
            window_size=config.TEACHER_WINDOW_SIZE,
            patch_size=config.INFERENCE_PATCH_SIZE,
            overlap=config.INFERENCE_OVERLAP
        )
        self.evaluator = ModelEvaluator(self.device)
        
        # Load models
        self.teacher_model = ModelLoader.load_teacher(self.device, config.TEACHER_MODEL_PATH)
        self.teacher_model.eval()
        
        self.student_model = ModelLoader.create_student_model()
        self.student_model.to(self.device)

        # Setup loss functions
        self.mse_loss = nn.MSELoss()
        if lpips is not None:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
        else:
            self.lpips_loss = None

        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)

        # Training state
        self.current_epoch = 0
        self.best_psnr = 0
        self.loss_weights = {'mse': config.MSE_WEIGHT, 'lpips': config.LPIPS_WEIGHT}

    def setup_datasets(self):
        """Setup training and validation datasets"""
        # Training dataset
        train_dataset = SuperResolutionDataset(
            self.config.TRAIN_HR_FOLDER,
            patch_size=self.config.PATCH_SIZE,
            scale_factor=self.config.SCALE_FACTOR,
            is_training=True
        )
        
        # Validation dataset
        val_dataset = SuperResolutionDataset(
            self.config.VAL_HR_FOLDER,
            patch_size=self.config.PATCH_SIZE,
            scale_factor=self.config.SCALE_FACTOR,
            is_training=False
        )
        
        # Create dataloaders
        self.train_loader = create_dataloader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

    def compute_loss(self, student_output, teacher_output, hr_target):
        """Compute combined loss for knowledge distillation"""
        # MSE loss against ground truth
        mse_loss_val = self.mse_loss(student_output, hr_target)
        
        # Knowledge distillation loss (MSE between student and teacher)
        kd_loss = self.mse_loss(student_output, teacher_output)
        
        # LPIPS loss (if available)
        lpips_loss_val = 0
        if self.lpips_loss is not None:
            # Convert to [-1, 1] range for LPIPS
            student_lpips = student_output * 2.0 - 1.0
            hr_lpips = hr_target * 2.0 - 1.0
            lpips_loss_val = self.lpips_loss(student_lpips, hr_lpips).mean()
        
        # Combined loss
        total_loss = (self.loss_weights['mse'] * mse_loss_val + 
                     0.5 * kd_loss + 
                     self.loss_weights['lpips'] * lpips_loss_val)
        
        return {
            'total': total_loss,
            'mse': mse_loss_val,
            'kd': kd_loss,
            'lpips': lpips_loss_val
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.student_model.train()
        total_losses = {'total': 0, 'mse': 0, 'kd': 0, 'lpips': 0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for lr_batch, hr_batch in pbar:
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)

            # Get teacher output (no gradients)
            with torch.no_grad():
                teacher_output = self.inference_processor.get_teacher_output(
                    lr_batch, self.teacher_model, use_forward_func=True
                )

            # Student forward pass
            student_output = self.student_model(lr_batch)

            # Compute loss
            losses = self.compute_loss(student_output, teacher_output, hr_batch)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()

            # Update running averages
            for key, loss_val in losses.items():
                if isinstance(loss_val, torch.Tensor):
                    total_losses[key] += loss_val.item()
                else:
                    total_losses[key] += loss_val
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Total': f"{losses['total'].item():.4f}",
                'MSE': f"{losses['mse'].item():.4f}",
                'KD': f"{losses['kd'].item():.4f}"
            })

        # Calculate average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        return avg_losses

    def validate(self):
        """Validate the model"""
        self.student_model.eval()
        
        # Use a subset for faster validation
        val_subset = torch.utils.data.Subset(self.val_loader.dataset, range(0, min(100, len(self.val_loader.dataset))))
        
        avg_psnr, avg_lpips = self.evaluator.evaluate_model(
            self.student_model, 
            val_subset, 
            "Student"
        )
        
        return avg_psnr, avg_lpips

    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'psnr': psnr,
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.config.STUDENT_CHECKPOINT_PATH)
            self.logger.info(f"New best model saved with PSNR: {psnr:.2f}")

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            train_losses = self.train_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log training losses
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} - "
                f"Total Loss: {train_losses['total']:.4f}, "
                f"MSE: {train_losses['mse']:.4f}, "
                f"KD: {train_losses['kd']:.4f}"
            )
            
            # Validation
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_psnr, val_lpips = self.validate()
                
                # Save checkpoint
                is_best = val_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = val_psnr
                
                if self.config.SAVE_BEST_ONLY:
                    if is_best:
                        self.save_checkpoint(epoch, val_psnr, is_best=True)
                else:
                    self.save_checkpoint(epoch, val_psnr, is_best)
        
        self.logger.info(f"Training completed. Best PSNR: {self.best_psnr:.2f}")


def main():
    """Main training function"""
    trainer = SuperResolutionTrainer()
    trainer.setup_datasets()
    trainer.train()


if __name__ == "__main__":
    main()

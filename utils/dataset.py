"""
Dataset utilities for super resolution training and evaluation.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import os


class SuperResolutionDataset(Dataset):
    """Dataset for super resolution training and evaluation"""

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


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """Create a dataloader with common settings"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

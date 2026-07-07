"""Verification script to test the preprocessing and augmentation pipeline with synthetic data."""
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
from src.preprocessing.image_preprocessor import (
    skull_strip, resize_image, apply_clahe, denoise_image, normalize_image
)
from src.preprocessing.augmentation import (
    get_training_augmentation, get_validation_augmentation
)

def test_pipeline():
    # Create a synthetic 512x512 grayscale image with a bright circle (fake brain)
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(img, (256, 256), 150, 200, -1)
    cv2.circle(img, (280, 240), 30, 255, -1)  # fake tumor
    print(f"1. Original shape: {img.shape}, dtype: {img.dtype}")

    img_ss = skull_strip(img)
    print(f"2. Skull stripped shape: {img_ss.shape}")

    img_resized = resize_image(img_ss, (224, 224))
    print(f"3. Resized shape: {img_resized.shape}")

    img_clahe = apply_clahe(img_resized)
    print(f"4. CLAHE shape: {img_clahe.shape}")

    img_denoised = denoise_image(img_clahe)
    print(f"5. Denoised shape: {img_denoised.shape}")

    img_norm = normalize_image(img_denoised)
    print(f"6. Normalized shape: {img_norm.shape}, range: [{img_norm.min():.3f}, {img_norm.max():.3f}]")

    # Verify normalization range
    assert img_norm.min() >= 0.0 and img_norm.max() <= 1.0, "Normalization out of range!"
    assert img_norm.dtype == np.float32, "Normalization dtype should be float32!"
    assert img_resized.shape == (224, 224), "Resize failed!"

    # Test augmentation pipelines
    transform = get_training_augmentation()
    aug = transform(image=img_denoised)
    aug_img = aug["image"]
    print(f"7. Training augmented tensor shape: {aug_img.shape}, type: {type(aug_img)}")
    assert isinstance(aug_img, torch.Tensor), "Training augmentation should return a tensor!"

    val_transform = get_validation_augmentation()
    val_aug = val_transform(image=img_denoised)
    val_img = val_aug["image"]
    print(f"8. Validation tensor shape: {val_img.shape}, type: {type(val_img)}")
    assert isinstance(val_img, torch.Tensor), "Validation augmentation should return a tensor!"

    print("\n=== All preprocessing and augmentation tests PASSED! ===")

if __name__ == "__main__":
    test_pipeline()

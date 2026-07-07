import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_training_augmentation():
    """
    Returns an albumentations transform pipeline for training.
    Includes rotation, flips, brightness/contrast, and elastic deformation.
    """
    return A.Compose([
        A.Rotate(limit=15, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        # Elastic deformation is very useful for medical images
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        ToTensorV2()
    ])

def get_validation_augmentation():
    """
    Returns an albumentations transform pipeline for validation/testing.
    Only converts to PyTorch tensor.
    """
    return A.Compose([
        ToTensorV2()
    ])

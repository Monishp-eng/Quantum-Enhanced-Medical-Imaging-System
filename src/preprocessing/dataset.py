import os
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from .image_preprocessor import preprocess_image
from .augmentation import get_training_augmentation, get_validation_augmentation

class BrainTumorDataset(Dataset):
    """PyTorch Dataset for Brain Tumor MRI."""
    def __init__(self, image_paths, labels, transform=None, grayscale=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Preprocess: load, skull strip, resize, clahe, denoise, normalize
        img = preprocess_image(path, grayscale=self.grayscale)
        
        # Albumentations expects (H, W, C) or (H, W) if grayscale
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        # If no transform or to_tensor used, ensure it is a tensor
        if not isinstance(img, torch.Tensor):
            # Convert to (C, H, W)
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                
        # Albumentations ToTensorV2 converts to (C, H, W) but we need to ensure type
        img = img.float()
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

def get_dataloaders(data_dir, batch_size=32, num_workers=0, grayscale=True):
    """
    Factory function returning train, val, and test loaders.
    Implements 70/15/15 stratified split and class-weighted sampling for training.
    """
    data_dir = Path(data_dir)
    
    # Look inside Training or Testing to find the actual class directories
    search_dir = data_dir / "Training"
    if not search_dir.exists():
        search_dir = data_dir / "Testing"
    if not search_dir.exists():
        search_dir = data_dir
        
    classes = sorted([d.name for d in search_dir.iterdir() if d.is_dir() and d.name not in ["Training", "Testing"]])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    
    # It seems the kaggle dataset has 'Training' and 'Testing' folders.
    # We will combine them and do our own 70/15/15 split to meet requirements.
    for split_dir in ["Training", "Testing"]:
        split_path = data_dir / split_dir
        if not split_path.exists():
            continue
        for cls_name in classes:
            cls_dir = split_path / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[cls_name])
                
    if not all_paths:
        raise ValueError(f"No images found in {data_dir}. Have you downloaded the dataset?")

    # 70/15/15 split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    # Class-weighted sampling
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_dataset = BrainTumorDataset(
        train_paths, train_labels, 
        transform=get_training_augmentation(), 
        grayscale=grayscale
    )
    val_dataset = BrainTumorDataset(
        val_paths, val_labels, 
        transform=get_validation_augmentation(), 
        grayscale=grayscale
    )
    test_dataset = BrainTumorDataset(
        test_paths, test_labels, 
        transform=get_validation_augmentation(), 
        grayscale=grayscale
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, class_to_idx

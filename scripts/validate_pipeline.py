import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.dataset import get_dataloaders

def validate():
    data_dir = Path("data/raw")
    if not data_dir.exists() or not list(data_dir.glob("*/*/*.jpg")):
        print("Data directory is empty or missing. Please run download_data.py first.")
        return

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, class_mapping = get_dataloaders(data_dir, batch_size=16)

    print(f"Class mapping: {class_mapping}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    print("\nValidating Training Loader...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image tensor type: {images.dtype}")
        print(f"Label tensor type: {labels.dtype}")
        # Assuming grayscale images: [batch_size, 1, 224, 224]
        assert images.shape[1:] == (1, 224, 224), f"Unexpected image shape: {images.shape}"
        break
        
    print("\nPipeline Validation Successful!")

if __name__ == "__main__":
    validate()

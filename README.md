# Quantum-Enhanced Medical Imaging System

This repository contains the source code for the QT-2.21 Quantum-Enhanced Medical Imaging for Cancer Detection project.

## Data Pipeline setup (Loki's Assignment)

### Prerequisites

You need Python 3.9+ and pip installed. We recommend using a virtual environment.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Kaggle Credentials:**
    The dataset is downloaded using the Kaggle API. You must have your `kaggle.json` configured.
    - Go to your Kaggle account settings and click "Create New API Token".
    - Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<User>\.kaggle\kaggle.json` (Windows).
    - Ensure it has the correct permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux).

### Downloading the Dataset

Run the provided script to automatically download and extract the Brain Tumor MRI Dataset into `data/raw/`:
```bash
python scripts/download_data.py
```

### Exploratory Data Analysis (EDA)

Explore the dataset distribution and sample images using the EDA notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Preprocessing Demo

To see a step-by-step visual demonstration of the preprocessing pipeline (Skull stripping, Resizing, CLAHE, Denoising, Normalization, and Data Augmentation):
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Using the PyTorch DataLoaders

The core dataset functionality is provided in `src/preprocessing/dataset.py`.

```python
from src.preprocessing.dataset import get_dataloaders

# Retrieve DataLoaders with stratified 70/15/15 split
# Training loader includes random augmentations and class-weighted sampling
train_loader, val_loader, test_loader, class_mapping = get_dataloaders('data/raw')

for images, labels in train_loader:
    print(images.shape) # e.g., (32, 1, 224, 224)
    print(labels.shape) # e.g., (32,)
    break
```

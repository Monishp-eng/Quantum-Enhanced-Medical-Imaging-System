import os
import subprocess
from pathlib import Path

def download_brain_tumor_dataset():
    """
    Downloads the Brain Tumor MRI Dataset from Kaggle and extracts it to data/raw/
    Dataset: masoudnickparvar/brain-tumor-mri-dataset
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    
    print(f"Setting up dataset at: {raw_data_dir}")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    dataset_name = "masoudnickparvar/brain-tumor-mri-dataset"
    
    try:
        print(f"Downloading {dataset_name} using kaggle CLI...")
        # Use subprocess to run the kaggle command
        # This requires kaggle to be installed and credentials configured
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name,
            "-p", str(raw_data_dir), "--unzip"
        ], check=True)
        print("Dataset downloaded and extracted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure Kaggle CLI is installed and configured: {e}")
        print("You can manually download it from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it using `pip install kaggle` and set up your ~/.kaggle/kaggle.json credentials.")

if __name__ == "__main__":
    download_brain_tumor_dataset()

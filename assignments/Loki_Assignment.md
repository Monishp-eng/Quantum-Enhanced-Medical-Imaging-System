# 🟢 Assignment Sheet — Loki
## Role: Data Engineer & Preprocessing Lead

> **Project:** QT-2.21 — Quantum-Enhanced Medical Imaging for Cancer Detection  
> **Evaluation Criteria Owned:** Soundness of Preprocessing & Feature Engineering (30%) — *preprocessing half*

---

## Your Responsibilities

1. **Download and organize** the Brain Tumor MRI Dataset from Kaggle
2. Implement the full **image preprocessing pipeline**: resize, CLAHE contrast enhancement, normalization, Gaussian denoising, skull stripping
3. Build the **data augmentation** module: rotation, flipping, elastic deformation, brightness/contrast adjustment
4. Create the **PyTorch Dataset & DataLoader** classes with stratified train/val/test split (70/15/15)
5. Perform **Exploratory Data Analysis (EDA)**: class distribution, sample visualization, image statistics
6. Handle **class imbalance** using weighted sampling or oversampling
7. Write **setup & reproducibility documentation** and verify the end-to-end pipeline runs cleanly
8. Support SK with any data loading or format issues during feature extraction

---

## Files You Own

```
quantum-medical-imaging/
├── data/
│   ├── raw/                          ← Downloaded dataset
│   │   ├── Training/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── pituitary/
│   │   │   └── notumor/
│   │   └── Testing/
│   └── processed/                    ← Preprocessed images & splits
│       ├── train/
│       ├── val/
│       └── test/
├── src/preprocessing/
│   ├── __init__.py
│   ├── image_preprocessor.py          ← Resize, CLAHE, normalize, denoise
│   ├── augmentation.py                ← All augmentation transforms
│   └── dataset.py                     ← PyTorch Dataset + DataLoader factory
├── scripts/
│   └── download_data.py               ← Kaggle dataset download script
└── notebooks/
    ├── 01_eda.ipynb                   ← Exploratory Data Analysis
    └── 02_preprocessing.ipynb         ← Preprocessing walkthrough & demo
```

---

## Sprint-Wise Task Breakdown

### Sprint 1 — Foundation (Jul 8 – Jul 18)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 1 | Download Brain Tumor MRI Dataset from Kaggle and organize into `data/raw/` | Jul 9 | Dataset in `data/raw/` |
| 2 | Write `scripts/download_data.py` — automates download using Kaggle CLI or manual instructions | Jul 9 | Script file |
| 3 | **Exploratory Data Analysis** — class distribution bar chart, sample images grid (4×4), image size distribution, pixel intensity histograms | Jul 11 | `01_eda.ipynb` |
| 4 | Implement `image_preprocessor.py` with the following steps: | Jul 14 | Tested module |
|   | — Load image (grayscale or RGB based on config) | | |
|   | — Skull stripping (contour-based or thresholding) | | |
|   | — Resize to 224×224 | | |
|   | — CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8) | | |
|   | — Normalize to [0, 1] range | | |
|   | — Gaussian blur denoising (kernel=3×3) | | |
| 5 | Implement `augmentation.py` with transforms: | Jul 15 | Tested module |
|   | — Random rotation (±15°) | | |
|   | — Random horizontal flip (p=0.5) | | |
|   | — Random vertical flip (p=0.3) | | |
|   | — Random brightness/contrast adjustment | | |
|   | — Elastic deformation (medical imaging specific) | | |
|   | — Optional: Mixup / CutMix for training | | |
| 6 | Implement `dataset.py`: | Jul 16 | Tested module |
|   | — `BrainTumorDataset(torch.utils.data.Dataset)` class | | |
|   | — Stratified train/val/test split (70/15/15) | | |
|   | — `get_dataloaders()` factory function returning train/val/test loaders | | |
|   | — Class-weighted sampling for imbalanced classes | | |
| 7 | Write `02_preprocessing.ipynb` — visual demo of each preprocessing step | Jul 17 | Notebook |
| 8 | **Handoff to SK** — deliver preprocessed dataset + working DataLoaders | Jul 16 | ✅ Handoff done |
| 9 | **Sprint 1 Review** — present EDA findings, demonstrate preprocessing pipeline | Jul 18 | ✅ Review done |

### Sprint 2 — Core Development (Jul 19 – Jul 31)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 10 | Harden data pipeline — handle edge cases (corrupt images, wrong formats, empty folders) | Jul 21 | Robust pipeline |
| 11 | Add advanced augmentations if needed — CutOut, GridDistortion, CLAHE variations | Jul 22 | Updated `augmentation.py` |
| 12 | Support SK with data loading issues — debug DataLoader batching, memory optimization | Jul 19–28 | Ongoing support |
| 13 | Add data validation checks — verify all images load, correct label mapping, no data leakage between splits | Jul 25 | Validation script |
| 14 | Profile data pipeline performance — measure DataLoader throughput, identify bottlenecks | Jul 28 | Performance notes |
| 15 | **Sprint 2 Review** — pipeline stable, no data bugs | Jul 31 | ✅ Review done |

### Sprint 3 — Polish & Report (Aug 1 – Aug 10)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 16 | Write setup documentation — environment setup, dataset download instructions, dependencies | Aug 2 | Section in README |
| 17 | Run end-to-end **reproducibility test** — clone repo fresh, follow README, verify pipeline runs | Aug 4 | ✅ Reproducibility verified |
| 18 | Provide dataset statistics section to Akshya for technical report | Aug 5 | Dataset stats document |
| 19 | Final code cleanup — docstrings, type hints, remove debug prints | Aug 7 | Clean code |

---

## Key Technical Specifications

### Preprocessing Pipeline (step-by-step)

```python
# What your image_preprocessor.py should do:

1. cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # Load as grayscale
2. skull_strip(img)                          # Remove non-brain regions
   → Find largest contour, create mask, apply
3. cv2.resize(img, (224, 224))               # Uniform size
4. cv2.createCLAHE(clipLimit=2.0,            # Contrast enhancement
      tileGridSize=(8,8)).apply(img)
5. img.astype(np.float32) / 255.0            # Normalize to [0,1]
6. cv2.GaussianBlur(img, (3,3), 0)           # Denoise
```

### Dataset Class Structure

```python
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        ...
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        # Load, preprocess, augment, return (tensor, label)
        ...

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # Returns: train_loader, val_loader, test_loader
    ...
```

### EDA Checklist (for `01_eda.ipynb`)

- [ ] Total number of images per class (bar chart)
- [ ] Sample images grid — 4 random images per class
- [ ] Image size distribution (histogram)
- [ ] Pixel intensity distribution per class
- [ ] Class imbalance ratio
- [ ] Corrupt/unreadable image check

---

## What You Deliver To Others

| To | What | Format | When |
|----|------|--------|------|
| **SK** | Preprocessed images + working `dataset.py` with DataLoaders | `data/processed/`, `dataset.py` | Jul 16 |
| **Akshya** | Dataset statistics for technical report | Text/table | Aug 5 |

## What You Receive From Others

| From | What | When |
|------|------|------|
| **Monish** | `config.yaml` with data paths and split ratios | Jul 9 |

---

## Deliverables Checklist

- [ ] Dataset downloaded and organized in `data/raw/`
- [ ] `download_data.py` script working
- [ ] `01_eda.ipynb` — complete EDA with visualizations
- [ ] `image_preprocessor.py` — all preprocessing steps tested
- [ ] `augmentation.py` — all augmentation transforms tested
- [ ] `dataset.py` — Dataset class + DataLoader factory working
- [ ] `02_preprocessing.ipynb` — visual demo of pipeline
- [ ] Stratified split verified (no data leakage)
- [ ] Class-weighted sampling implemented
- [ ] End-to-end reproducibility verified
- [ ] Setup documentation written

---

## Git Branch

```
feature/data-pipeline
```

All your work goes into this branch. Create a **Pull Request** to `dev` when ready. Monish will review.

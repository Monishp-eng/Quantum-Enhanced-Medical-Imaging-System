# 🟠 Assignment Sheet — SK
## Role: Feature Engineer & Model Training Lead

> **Project:** QT-2.21 — Quantum-Enhanced Medical Imaging for Cancer Detection  
> **Evaluation Criteria Owned:** Soundness of Feature Engineering (30%) — *feature half* + Classification Performance with Emphasis on Sensitivity (25%)

---

## Your Responsibilities

1. Implement **CNN-based feature extraction** using pretrained ResNet50 (2048-dim feature vectors)
2. Implement **handcrafted feature extraction**: GLCM textures, HOG descriptors, intensity statistics, wavelet coefficients
3. Build the **feature fusion pipeline** — concatenate CNN + handcrafted features into a unified matrix
4. Implement **classical feature selection baselines**: Mutual Information, Recursive Feature Elimination (RFE), PCA, SelectKBest
5. Implement **classical hyperparameter tuning baselines**: GridSearchCV, RandomSearchCV, Optuna (TPE)
6. Train all **classification models**: SVM (RBF), XGBoost, Random Forest, fine-tuned ResNet50
7. Deliver trained model checkpoints and test predictions to Akshya for evaluation

---

## Files You Own

```
quantum-medical-imaging/
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── cnn_features.py             ← ResNet50 feature extraction
│   │   ├── handcrafted.py              ← GLCM, HOG, intensity, wavelets
│   │   └── feature_combiner.py         ← Merge all features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_classifier.py           ← ResNet50 fine-tuned classifier
│   │   ├── ml_classifier.py            ← SVM, XGBoost, Random Forest
│   │   └── ensemble.py                 ← Optional ensemble model
│   └── optimization/
│       ├── classical_feature_selection.py  ← MI, RFE, PCA, SelectKBest
│       └── classical_hyperparam.py         ← GridSearch, Optuna
└── notebooks/
    ├── 03_feature_engineering.ipynb     ← Feature extraction walkthrough
    └── 05_model_training.ipynb          ← Training experiments
```

---

## Sprint-Wise Task Breakdown

### Sprint 1 — Foundation (Jul 8 – Jul 18)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 1 | Implement **`cnn_features.py`** — ResNet50 feature extractor | Jul 12 | Tested module |
|   | — Load pretrained ResNet50 (ImageNet weights) | | |
|   | — Remove final FC layer | | |
|   | — Extract 2048-dim feature vector from `avgpool` layer | | |
|   | — Handle batch processing for full dataset | | |
| 2 | Implement **`handcrafted.py`** — all handcrafted feature extractors | Jul 15 | Tested module |
|   | — **GLCM features**: contrast, correlation, energy, homogeneity, dissimilarity (at 4 angles × 5 distances = ~20 features) | | |
|   | — **HOG features**: Histogram of Oriented Gradients (cell_size=16, ~36 features) | | |
|   | — **Intensity statistics**: mean, std, skewness, kurtosis, entropy, percentiles (~10 features) | | |
|   | — **Wavelet features**: DWT coefficients using Haar/Daubechies (4 levels, ~32 features) | | |
| 3 | Implement **`feature_combiner.py`** — merge all feature types | Jul 16 | Tested module |
|   | — Concatenate: CNN (2048) + GLCM (20) + HOG (36) + Intensity (10) + Wavelet (32) = ~2146 features | | |
|   | — StandardScaler normalization | | |
|   | — Save as `features.npy` and `labels.npy` | | |
| 4 | Write `03_feature_engineering.ipynb` — demonstrate each feature type with visualizations | Jul 17 | Notebook |
| 5 | **Sprint 1 Review** — feature extraction working, sample outputs verified | Jul 18 | ✅ Review done |

### Sprint 2 — Core Development (Jul 19 – Jul 31)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 6 | Run feature extraction on full preprocessed dataset → produce `features.npy`, `labels.npy` | Jul 20 | Feature files |
| 7 | **Handoff to Monish** — deliver combined feature matrix for QIEO optimization | Jul 22 | ✅ Handoff done |
| 8 | Implement **`classical_feature_selection.py`** — 4 baseline methods: | Jul 24 | Tested module |
|   | — Mutual Information (`mutual_info_classif`) | | |
|   | — Recursive Feature Elimination (RFE with SVM) | | |
|   | — PCA (retain 95% variance) | | |
|   | — SelectKBest (f_classif, k=100) | | |
| 9 | Implement **`classical_hyperparam.py`** — 3 baseline methods: | Jul 26 | Tested module |
|   | — GridSearchCV (exhaustive, small grid) | | |
|   | — RandomizedSearchCV (100 iterations) | | |
|   | — Optuna with TPE sampler (200 trials) | | |
| 10 | **Receive from Monish** — `selected_features.json` + `best_params.json` | Jul 28 | ✅ Received |
| 11 | Train **all 6 model configurations**: | Jul 31 | Model checkpoints |
|   | — **Baseline A**: MI/RFE features + GridSearch + SVM | | |
|   | — **Baseline B**: PCA/SelectKBest features + Optuna + XGBoost | | |
|   | — **Baseline C**: All features + manual tuning + ResNet50 (fine-tuned) | | |
|   | — **Quantum A**: QIEO features + GridSearch + SVM | | |
|   | — **Quantum B**: QIEO features + QIEO hyperparams + XGBoost | | |
|   | — **Quantum C**: QIEO features + QIEO hyperparams + ResNet50 | | |
| 12 | **Sprint 2 Review** — all models trained, initial accuracy numbers available | Jul 31 | ✅ Review done |

### Sprint 3 — Polish & Report (Aug 1 – Aug 10)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 13 | Implement **`cnn_classifier.py`** fine-tuning details: | Aug 3 | Trained CNN |
|    | — Freeze early ResNet50 layers, train last ~20 params | | |
|    | — Custom head: Dropout(0.5) → Linear(2048,512) → ReLU → BN → Dropout(0.3) → Linear(512,4) | | |
|    | — Focal Loss for class imbalance (emphasis on sensitivity) | | |
|    | — AdamW optimizer + Cosine Annealing LR scheduler | | |
|    | — Early stopping (patience=10 on val loss) | | |
| 14 | Model fine-tuning and ablation studies — test different feature subsets, compare | Aug 5 | Ablation results |
| 15 | **Handoff to Akshya** — deliver all model checkpoints + test set predictions | Aug 2 | ✅ Handoff done |
|    | — `.pth` files for CNN models | | |
|    | — `.pkl` files for SVM/XGBoost models | | |
|    | — `predictions.csv` with columns: [image_id, true_label, predicted_label, probabilities] | | |
| 16 | Write `05_model_training.ipynb` — document training process, loss curves, learning rates | Aug 6 | Notebook |
| 17 | Final code cleanup — docstrings, type hints | Aug 8 | Clean code |

---

## Key Technical Specifications

### CNN Feature Extraction

```python
class CNNFeatureExtractor:
    def __init__(self):
        model = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(model.children())[:-1])
        self.extractor.eval()

    def extract(self, image_tensor):  # Input: (B, 3, 224, 224)
        with torch.no_grad():
            features = self.extractor(image_tensor)
        return features.squeeze()    # Output: (B, 2048)
```

### Handcrafted Features Summary

| Feature | Library | Function | Output Dims |
|---------|---------|----------|-------------|
| GLCM | `skimage.feature` | `graycomatrix()` + `graycoprops()` | ~20 |
| HOG | `skimage.feature` | `hog()` | ~36 |
| Intensity | `numpy`, `scipy.stats` | `mean, std, skew, kurtosis, entropy` | ~10 |
| Wavelet | `pywt` | `wavedec2()` (Haar, 4 levels) | ~32 |

### 6 Model Configurations to Train

| Config | Features | Hyperparams | Classifier | Your Priority |
|--------|----------|-------------|------------|:---:|
| Baseline A | MI / RFE selected | GridSearchCV | SVM (RBF) | ⭐⭐ |
| Baseline B | PCA / SelectKBest | Optuna (TPE) | XGBoost | ⭐⭐⭐ |
| Baseline C | All 2146 features | Manual tuning | ResNet50 fine-tuned | ⭐⭐⭐ |
| Quantum A | QIEO selected | GridSearchCV | SVM (RBF) | ⭐⭐ |
| Quantum B | QIEO selected | QIEO optimized | XGBoost | ⭐⭐⭐ |
| Quantum C | QIEO selected | QIEO optimized | ResNet50 fine-tuned | ⭐⭐⭐ |

---

## What You Receive From Others

| From | What | Format | When |
|------|------|--------|------|
| **Loki** | Preprocessed images + DataLoaders | `data/processed/`, `dataset.py` | Jul 16 |
| **Monish** | QIEO-selected feature indices | `selected_features.json` | Jul 28 |
| **Monish** | QIEO-optimized hyperparameters | `best_params.json` | Jul 28 |

## What You Deliver To Others

| To | What | Format | When |
|----|------|--------|------|
| **Monish** | Combined feature matrix | `features.npy`, `labels.npy` | Jul 22 |
| **Akshya** | All trained model checkpoints | `.pth` / `.pkl` files | Aug 2 |
| **Akshya** | Test set predictions for all 6 configs | `predictions.csv` per config | Aug 2 |

---

## Deliverables Checklist

- [ ] `cnn_features.py` — ResNet50 extraction producing 2048-dim vectors
- [ ] `handcrafted.py` — GLCM, HOG, intensity, wavelet features
- [ ] `feature_combiner.py` — fusion + normalization + save
- [ ] `features.npy` + `labels.npy` — full dataset feature matrix
- [ ] `classical_feature_selection.py` — MI, RFE, PCA, SelectKBest
- [ ] `classical_hyperparam.py` — GridSearch, RandomSearch, Optuna
- [ ] `ml_classifier.py` — SVM, XGBoost, Random Forest trained
- [ ] `cnn_classifier.py` — ResNet50 fine-tuned with Focal Loss
- [ ] All 6 model configurations trained and checkpoints saved
- [ ] `predictions.csv` for all 6 configurations
- [ ] `03_feature_engineering.ipynb` complete
- [ ] `05_model_training.ipynb` complete

---

## Git Branch

```
feature/feature-engineering    (Sprint 1)
feature/model-training         (Sprint 2-3)
```

All your work goes into these branches. Create **Pull Requests** to `dev` when ready. Monish will review.

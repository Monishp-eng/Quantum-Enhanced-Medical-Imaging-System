# QT-2.21: Quantum-Enhanced Medical Imaging for Cancer Detection

A comprehensive Brain Tumor classification system from MRI scans that leverages **Quantum-Inspired Evolutionary Optimization (QIEO)** for feature selection and hyperparameter tuning, and compares performance against classically optimized machine learning baselines.

**Team**: Monish (Quantum Optimization) · Loki (Data Engineering) · SK (Feature Engineering & Models) · Akshya (Evaluation & Explainability)

---

## 🏗️ Architecture

```
Raw MRI Dataset (Kaggle)
  │
  ├── Phase 1: Preprocessing Pipeline
  │     Skull Strip → Resize (224×224) → CLAHE → Denoise → Normalize → Augment
  │
  ├── Phase 2: Feature Engineering
  │     ResNet50 Deep Features (2048-dim)
  │     + GLCM · HOG · Intensity · DWT Wavelets (98-dim)
  │     = Fused Feature Vector (2146-dim)
  │
  ├── Phase 3: Optimization & Training (6 Configurations)
  │     Baseline A: MI Selection + SVM (GridSearchCV)
  │     Baseline B: SelectKBest + XGBoost (Optuna)
  │     Baseline C: Fine-tuned ResNet50 (Focal Loss)
  │     Quantum A:  QIEO Feature Selection + SVM
  │     Quantum B:  QIEO Feature Selection + XGBoost (QIEO Hyperparams)
  │     Quantum C:  QIEO-tuned ResNet50
  │
  └── Phase 4: Evaluation & Explainability
        Accuracy · Sensitivity · Specificity · Precision · F1 · AUC-ROC
        McNemar's Statistical Test · Confusion Matrices · ROC/PR Curves
        Grad-CAM Heatmaps · Saliency Maps · Attention Overlays
```

---

## 📁 Project Structure

```
├── config/
│   └── config.yaml              # Centralized pipeline configuration
├── data/
│   └── raw/                     # Brain Tumor MRI dataset (4 classes)
├── models/                      # Saved model checkpoints & predictions
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Preprocessing pipeline walkthrough
│   ├── 03_feature_engineering.ipynb  # Feature extraction & fusion demo
│   ├── 05_model_training.ipynb  # Training all 6 configurations
│   ├── 06_evaluation.ipynb      # Evaluation & comparison
│   └── 07_explainability.ipynb  # Grad-CAM & Saliency maps
├── reports/
│   ├── figures/                 # Confusion matrices, ROC/PR curves
│   │   └── explainability/      # Grad-CAM & Saliency overlays
│   ├── results/                 # JSON metrics & comparison CSV
│   └── technical_report.md      # Full technical report
├── scripts/
│   ├── download_data.py         # Kaggle dataset downloader
│   ├── generate_mock_data.py    # Synthetic data for quick testing
│   ├── train.py                 # End-to-end training orchestrator
│   ├── evaluate.py              # Evaluation & explainability generator
│   ├── test_pipeline.py         # Preprocessing unit tests
│   └── validate_pipeline.py     # DataLoader validation
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessor.py    # Skull strip, CLAHE, resize, normalize
│   │   ├── augmentation.py          # Albumentations pipeline
│   │   └── dataset.py               # PyTorch Dataset & DataLoader
│   ├── features/
│   │   ├── cnn_features.py          # ResNet50 feature extractor
│   │   ├── handcrafted.py           # GLCM, HOG, Intensity, Wavelet
│   │   └── feature_combiner.py      # Fusion & StandardScaler
│   ├── optimization/
│   │   ├── qieo_solver.py           # QIEO algorithm (Q-bits + rotation gates)
│   │   ├── quantum_feature_selection.py   # Binary feature mask optimization
│   │   ├── quantum_hyperparam.py          # Binary-encoded hyperparameter search
│   │   ├── classical_feature_selection.py # MI, RFE, PCA, SelectKBest
│   │   └── classical_hyperparam.py        # GridSearch, Optuna
│   ├── models/
│   │   ├── ml_classifier.py        # SVM, XGBoost, Random Forest wrappers
│   │   └── cnn_classifier.py       # ResNet50 fine-tuning with Focal Loss
│   ├── evaluation/
│   │   ├── metrics.py              # All classification metrics
│   │   ├── visualizations.py       # Plotting utilities
│   │   └── comparison.py           # McNemar's test & comparison tables
│   └── explainability/
│       ├── gradcam.py              # Grad-CAM implementation
│       ├── saliency.py             # Vanilla Saliency maps
│       └── attention_viz.py        # Heatmap overlay & comparison plots
├── assignments/                 # Team member assignment documents
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

**Option A — Kaggle API** (requires `~/.kaggle/kaggle.json`):
```bash
python scripts/download_data.py
```

**Option B — Synthetic Mock Data** (instant, no credentials needed):
```bash
python scripts/generate_mock_data.py
```

### 4. Run the Full Pipeline

**Training** (all 6 configurations):
```bash
python scripts/train.py
```

For a quick test run with a small data subset:
```bash
python scripts/train.py --quick-test
```

**Evaluation & Explainability**:
```bash
python scripts/evaluate.py
```

### 5. Explore Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Dataset distribution, class balance, sample visualization |
| `02_preprocessing.ipynb` | Step-by-step preprocessing pipeline demo |
| `03_feature_engineering.ipynb` | CNN + handcrafted feature extraction & fusion |
| `05_model_training.ipynb` | Training all 6 model configurations |
| `06_evaluation.ipynb` | Metrics, comparison tables, statistical tests |
| `07_explainability.ipynb` | Grad-CAM and Saliency map generation |

---

## 🧪 Model Configurations

| Config | Feature Selection | Classifier | Hyperparameter Tuning |
|--------|------------------|------------|----------------------|
| **Baseline A** | Mutual Information (top 100) | SVM (RBF) | GridSearchCV |
| **Baseline B** | SelectKBest (top 100) | XGBoost | Optuna TPE |
| **Baseline C** | End-to-end CNN | ResNet50 | Manual LR schedule |
| **Quantum A** | **QIEO** binary mask | SVM (RBF) | GridSearchCV |
| **Quantum B** | **QIEO** binary mask | XGBoost | **QIEO** binary-encoded |
| **Quantum C** | End-to-end CNN | ResNet50 | **QIEO**-derived settings |

---

## 📊 Evaluation Metrics

All models are evaluated on:
- **Accuracy** — Overall correctness
- **Sensitivity** (Recall / TPR) — Per-class and macro; critical for cancer detection
- **Specificity** (TNR) — Per-class and macro
- **Precision** — Macro averaged
- **F1-Score** — Weighted and macro
- **AUC-ROC** — Multiclass One-vs-Rest
- **Confusion Matrix** — Per-configuration heatmaps
- **McNemar's Test** — Statistical significance of Baseline vs. Quantum differences

---

## 🔬 Explainability (Stretch Goal)

Three XAI methods are implemented to highlight which image regions drive predictions:

| Method | Description |
|--------|-------------|
| **Grad-CAM** | Weighted activation maps from ResNet50 `layer4` |
| **Saliency Maps** | Absolute input gradients of target class score |
| **Attention Overlay** | JET colormap overlay on original MRI scans |

Generated for all 4 classes: Glioma, Meningioma, Pituitary, No Tumor.

---

## ⚙️ Configuration

All pipeline parameters are centralized in [`config/config.yaml`](config/config.yaml):
- Data paths, image sizes, split ratios
- Preprocessing settings (CLAHE, denoising)
- Augmentation parameters
- Feature extraction dimensions
- QIEO solver settings (population size, iterations, theta_max)
- Training hyperparameters
- Evaluation metrics and significance thresholds

---

## 📝 Technical Report

The full technical report is available at [`reports/technical_report.md`](reports/technical_report.md) and covers:
1. Executive Summary
2. Clinical Significance
3. Data Engineering & Preprocessing
4. Feature Engineering & Fusion Strategy
5. Model Configurations & Classifiers
6. QIEO Mathematical Formulation
7. Quantitative Results & Comparative Analysis
8. Model Explainability (Grad-CAM & Saliency)
9. Clinical Limitations & Future Directions
10. Conclusion

---

## 📜 License

This project is developed for the QT-2.21 academic challenge.

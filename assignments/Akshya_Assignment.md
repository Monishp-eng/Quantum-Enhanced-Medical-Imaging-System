# 🟣 Assignment Sheet — Akshya
## Role: Evaluation, Explainability & Documentation Lead

> **Project:** QT-2.21 — Quantum-Enhanced Medical Imaging for Cancer Detection  
> **Evaluation Criteria Owned:** Fairness & Completeness of Baseline Comparison (15%) + Clinical Limitations, Documentation & Presentation (10%)

---

## Your Responsibilities

1. Implement the complete **metrics module**: Accuracy, Sensitivity, Specificity, Precision, Recall, F1-Score, AUC-ROC
2. Generate **confusion matrices**, **ROC curves**, and **precision-recall curves** for all 6 model configurations
3. Build the **Classical vs. Quantum comparison** report with statistical tests and visual comparisons
4. Implement **Grad-CAM** and **Saliency Maps** for model explainability (stretch goal)
5. Create attention heatmap **overlay visualizations** on MRI scans
6. Write the full **Technical Report** covering methodology, results, and clinical limitations
7. Prepare the **final presentation slides**

---

## Files You Own

```
quantum-medical-imaging/
├── src/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  ← All metric computations
│   │   ├── comparison.py               ← Classical vs. Quantum comparison engine
│   │   └── visualizations.py           ← Confusion matrix, ROC, bar charts
│   └── explainability/
│       ├── __init__.py
│       ├── gradcam.py                  ← Grad-CAM implementation
│       ├── saliency.py                 ← Saliency map generation
│       └── attention_viz.py            ← Heatmap overlay on MRI scans
├── notebooks/
│   ├── 04_quantum_optimization.ipynb   ← (with Monish) Document QIEO results
│   ├── 06_evaluation.ipynb             ← Full evaluation walkthrough
│   └── 07_explainability.ipynb         ← Grad-CAM & saliency demo
├── reports/
│   ├── technical_report.md             ← Complete technical report
│   ├── figures/                        ← All generated plots & charts
│   └── results/                        ← CSV files of all metrics
└── scripts/
    └── evaluate.py                     ← Standalone evaluation script
```

---

## Sprint-Wise Task Breakdown

### Sprint 1 — Foundation (Jul 8 – Jul 18)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 1 | Implement **`metrics.py`** — all evaluation metrics: | Jul 12 | Tested module |
|   | — `compute_accuracy(y_true, y_pred)` | | |
|   | — `compute_sensitivity(y_true, y_pred)` (per-class & macro) | | |
|   | — `compute_specificity(y_true, y_pred)` (per-class & macro) | | |
|   | — `compute_precision(y_true, y_pred)` | | |
|   | — `compute_recall(y_true, y_pred)` | | |
|   | — `compute_f1(y_true, y_pred)` (weighted, macro, per-class) | | |
|   | — `compute_auc_roc(y_true, y_proba)` (one-vs-rest for multiclass) | | |
|   | — `compute_all_metrics(y_true, y_pred, y_proba)` → returns dict | | |
| 2 | Implement **`visualizations.py`** — plotting utilities: | Jul 15 | Tested module |
|   | — `plot_confusion_matrix(y_true, y_pred, class_names)` | | |
|   | — `plot_roc_curve(y_true, y_proba, class_names)` — multiclass one-vs-rest | | |
|   | — `plot_precision_recall_curve(y_true, y_proba)` | | |
|   | — `plot_metric_comparison_bar(results_dict)` — grouped bar chart | | |
|   | — `plot_training_curves(train_loss, val_loss, train_acc, val_acc)` | | |
|   | — All plots saved to `reports/figures/` as PNG (300 DPI) | | |
| 3 | Implement **`comparison.py`** scaffold — define comparison framework structure | Jul 17 | Skeleton code |
| 4 | **Sprint 1 Review** — metrics module tested with dummy data | Jul 18 | ✅ Review done |

### Sprint 2 — Core Development (Jul 19 – Jul 31)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 5 | Complete **`comparison.py`** — Classical vs. Quantum comparison engine: | Jul 25 | Tested module |
|   | — Load predictions from all 6 configurations | | |
|   | — Compute all metrics for each configuration | | |
|   | — Generate side-by-side comparison tables | | |
|   | — Statistical significance testing (McNemar's test or paired t-test on CV folds) | | |
|   | — Compute improvement percentages (Quantum over Classical) | | |
| 6 | Build `scripts/evaluate.py` — standalone evaluation script that: | Jul 27 | Working script |
|   | — Takes model checkpoint + test data as input | | |
|   | — Computes all metrics | | |
|   | — Generates all plots | | |
|   | — Saves results to `reports/results/` as CSV | | |
| 7 | Write `04_quantum_optimization.ipynb` (with Monish) — document QIEO feature selection results, convergence plots, selected feature analysis | Jul 29 | Notebook |
| 8 | **Sprint 2 Review** — comparison framework tested, ready for real model outputs | Jul 31 | ✅ Review done |

### Sprint 3 — Polish & Report (Aug 1 – Aug 10)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 9 | **Receive from SK** — all model checkpoints + `predictions.csv` for 6 configs | Aug 2 | ✅ Received |
| 10 | Run full evaluation on all 6 configurations — generate all metrics & plots | Aug 3 | Results in `reports/` |
| 11 | Implement **`gradcam.py`** (Stretch Goal): | Aug 4 | Tested module |
|    | — Register forward hooks on last conv layer of ResNet50 | | |
|    | — Register backward hooks to capture gradients | | |
|    | — Compute weighted activation maps | | |
|    | — Apply ReLU, upsample to input size, normalize [0,1] | | |
| 12 | Implement **`saliency.py`** (Stretch Goal): | Aug 5 | Tested module |
|    | — Enable input gradients | | |
|    | — Backprop target class score | | |
|    | — Take absolute value of input gradients | | |
|    | — Normalize and visualize | | |
| 13 | Implement **`attention_viz.py`** (Stretch Goal): | Aug 5 | Tested module |
|    | — Overlay Grad-CAM heatmap on original MRI (using cv2 colormap) | | |
|    | — Create side-by-side comparison: Original / Grad-CAM / Saliency | | |
|    | — Generate grid of examples for each class | | |
| 14 | Write `06_evaluation.ipynb` — full evaluation walkthrough with all plots | Aug 6 | Notebook |
| 15 | Write `07_explainability.ipynb` — Grad-CAM & saliency demo with sample MRIs | Aug 6 | Notebook |
| 16 | Write **`technical_report.md`** — complete technical report: | Aug 8 | Report |
|    | — *Section 1:* Introduction & Problem Statement | | |
|    | — *Section 2:* Dataset Description & EDA Summary (data from Loki) | | |
|    | — *Section 3:* Preprocessing & Feature Engineering Methodology | | |
|    | — *Section 4:* Quantum Optimization Methodology (input from Monish) | | |
|    | — *Section 5:* Classification Models & Training | | |
|    | — *Section 6:* Results & Performance Comparison | | |
|    | — *Section 7:* Explainability Analysis (Grad-CAM, Saliency) | | |
|    | — *Section 8:* Clinical Limitations & Ethical Considerations | | |
|    | — *Section 9:* Conclusion & Future Work | | |
|    | — *Appendix:* Full metrics tables, additional figures | | |
| 17 | Prepare **presentation slides** for final demo | Aug 9 | Slide deck |
| 18 | Final proofreading and formatting | Aug 10 | ✅ Submitted |

---

## Key Technical Specifications

### Metrics Module — Expected Output Format

```python
# compute_all_metrics() should return:
{
    "accuracy": 0.943,
    "sensitivity_macro": 0.931,
    "sensitivity_per_class": {"glioma": 0.92, "meningioma": 0.91, "pituitary": 0.97, "notumor": 0.95},
    "specificity_macro": 0.977,
    "precision_macro": 0.938,
    "recall_macro": 0.931,
    "f1_weighted": 0.942,
    "f1_macro": 0.934,
    "auc_roc_macro": 0.989,
    "auc_roc_per_class": {"glioma": 0.98, "meningioma": 0.97, "pituitary": 0.99, "notumor": 0.99}
}
```

### Comparison Table Format (for report)

| Metric | Baseline A | Baseline B | Baseline C | Quantum A | Quantum B | Quantum C |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Accuracy | — | — | — | — | — | — |
| Sensitivity | — | — | — | — | — | — |
| Specificity | — | — | — | — | — | — |
| Precision | — | — | — | — | — | — |
| F1-Score | — | — | — | — | — | — |
| AUC-ROC | — | — | — | — | — | — |
| **# Features Used** | — | — | — | — | — | — |
| **Training Time** | — | — | — | — | — | — |

### Grad-CAM Implementation Outline

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks on target_layer

    def generate(self, input_tensor, target_class):
        # 1. Forward pass → capture activations
        # 2. Backward pass → capture gradients
        # 3. weights = global_avg_pool(gradients)
        # 4. cam = ReLU(sum(weights * activations))
        # 5. Upsample to input size, normalize to [0, 1]
        return heatmap   # numpy array (224, 224)
```

### Technical Report Structure (Section Lengths)

| Section | Approx. Pages | Key Content |
|---------|:---:|-------------|
| 1. Introduction | 1 | Problem, motivation, objectives |
| 2. Dataset | 1 | Source, classes, statistics, limitations |
| 3. Preprocessing | 1.5 | Pipeline steps, augmentation, justification |
| 4. Quantum Optimization | 2 | QIEO algorithm, formulation, convergence |
| 5. Classification | 1.5 | Models, architectures, training strategy |
| 6. Results | 2.5 | All metrics, comparison tables, plots |
| 7. Explainability | 1.5 | Grad-CAM examples, saliency analysis |
| 8. Clinical Limitations | 1 | Bias, generalization, regulatory, ethics |
| 9. Conclusion | 0.5 | Summary, future directions |
| **Total** | **~12** | |

### Clinical Limitations Section — Points to Cover

- [ ] **Dataset bias** — single-source data, limited diversity in patient demographics
- [ ] **Generalization** — model trained on one MRI scanner/protocol may not transfer
- [ ] **Regulatory** — not FDA/CE approved, research use only
- [ ] **False negatives** — clinical risk of missing tumors
- [ ] **Explainability limitations** — Grad-CAM shows correlation, not causation
- [ ] **Data privacy** — de-identification, HIPAA considerations
- [ ] **Class imbalance** — impact on minority class performance

---

## What You Receive From Others

| From | What | Format | When |
|------|------|--------|------|
| **SK** | Trained model checkpoints | `.pth` / `.pkl` files | Aug 2 |
| **SK** | Test set predictions for all 6 configs | `predictions.csv` per config | Aug 2 |
| **Loki** | Dataset statistics for report | Text/table | Aug 5 |
| **Monish** | QIEO methodology notes for report | Text/notes | Aug 5 |
| **All** | Module docstrings & method descriptions | Inline docs | Aug 4 |

## What You Deliver To Others

| To | What | Format | When |
|----|------|--------|------|
| **All** | Final comparison results | `reports/results/` CSVs | Aug 6 |
| **All** | Technical report for review | `reports/technical_report.md` | Aug 8 |
| **All** | Presentation slides | Slide deck (PPT/Google Slides) | Aug 9 |

---

## Deliverables Checklist

- [ ] `metrics.py` — all metrics computed and tested
- [ ] `visualizations.py` — confusion matrix, ROC, PR curves, bar charts
- [ ] `comparison.py` — full Classical vs. Quantum comparison with statistical tests
- [ ] `evaluate.py` — standalone evaluation script
- [ ] `gradcam.py` — Grad-CAM heatmap generation (stretch)
- [ ] `saliency.py` — saliency map generation (stretch)
- [ ] `attention_viz.py` — heatmap overlay visualization (stretch)
- [ ] `06_evaluation.ipynb` — evaluation walkthrough notebook
- [ ] `07_explainability.ipynb` — explainability demo notebook
- [ ] `04_quantum_optimization.ipynb` — QIEO results documentation (with Monish)
- [ ] `reports/figures/` — all plots saved as PNG (300 DPI)
- [ ] `reports/results/` — metrics CSVs for all configurations
- [ ] `reports/technical_report.md` — complete 12-page technical report
- [ ] Presentation slides prepared
- [ ] Clinical limitations section thorough and honest

---

## Git Branch

```
feature/evaluation-xai
```

All your work goes into this branch. Create a **Pull Request** to `dev` when ready. Monish will review.

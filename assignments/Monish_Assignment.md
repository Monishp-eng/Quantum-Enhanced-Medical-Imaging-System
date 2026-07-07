# 🔷 Assignment Sheet — Monish
## Role: Project Lead & Quantum Optimization Engineer

> **Project:** QT-2.21 — Quantum-Enhanced Medical Imaging for Cancer Detection  
> **Evaluation Criteria Owned:** Effective Use of QuantumNow & Optimization Strategy (20%)

---

## Your Responsibilities

1. Define the **overall project architecture**, folder structure, and configuration
2. Implement **QIEO-based feature selection** — formulate feature selection as a binary combinatorial optimization problem and solve using quantum-inspired evolutionary optimization
3. Implement **QIEO-based hyperparameter optimization** — map classifier hyperparameters to a continuous search space and optimize via QIEO
4. If BQPhy SDK is not accessible, build a **QIEO simulator** that faithfully implements quantum probability amplitude encoding + quantum rotation gates
5. Write the master **training orchestrator** (`main.py`, `scripts/train.py`)
6. **Coordinate integration** — ensure all modules connect cleanly end-to-end
7. **Review all Pull Requests** before merge to `dev`
8. Own `requirements.txt`, `README.md`, and environment setup

---

## Files You Own

```
quantum-medical-imaging/
├── main.py                                    ← Master orchestrator
├── config/
│   └── config.yaml                            ← All paths, hyperparams, solver settings
├── requirements.txt                           ← Project dependencies
├── README.md                                  ← Setup & usage instructions
├── src/
│   └── optimization/
│       ├── __init__.py
│       ├── quantum_feature_selection.py        ← QIEO feature selection
│       └── quantum_hyperparam.py               ← QIEO hyperparameter tuning
└── scripts/
    └── train.py                               ← End-to-end training script
```

---

## Sprint-Wise Task Breakdown

### Sprint 1 — Foundation (Jul 8 – Jul 18)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 1 | Initialize Git repo, folder structure, `.gitignore` | Jul 8 | GitHub repo link |
| 2 | Create `config/config.yaml` with all paths, image size, split ratios, solver params | Jul 9 | `config.yaml` |
| 3 | Create `requirements.txt` with all dependencies | Jul 9 | `requirements.txt` |
| 4 | Research QIEO algorithm in depth — study quantum rotation gates, probability amplitude representation, population evolution | Jul 10–12 | Notes document |
| 5 | Design the QIEO solver interface — define `QIEOSolver` class API (inputs, outputs, config params) | Jul 13 | Class skeleton in code |
| 6 | **Sprint 1 Review** — verify Loki's data pipeline, SK's feature extractor scaffolds | Jul 18 | ✅ Review done |

### Sprint 2 — Core Development (Jul 19 – Jul 31)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 7 | Implement `QIEOSolver` core — population initialization with quantum amplitude encoding | Jul 20 | Working solver class |
| 8 | Implement quantum rotation gate updates + measurement (collapse to binary/continuous) | Jul 21 | Tested solver |
| 9 | Implement **`quantum_feature_selection.py`** — objective function = negative CV accuracy + sparsity penalty; binary mask optimization over ~2146 features | Jul 23 | `selected_features.json` |
| 10 | Implement **`quantum_hyperparam.py`** — continuous optimization over SVM (C, gamma) and XGBoost (7 hyperparams) search spaces | Jul 26 | `best_params.json` |
| 11 | **Handoff to SK** — deliver `selected_features.json` + `best_params.json` | Jul 28 | ✅ Handoff done |
| 12 | Build `main.py` — orchestrate full pipeline: preprocess → features → optimize → train → evaluate | Jul 29 | Working `main.py` |
| 13 | Integration testing — run full pipeline end-to-end on small data subset | Jul 31 | ✅ Pipeline runs |
| 14 | **Sprint 2 Review** — all modules integrated | Jul 31 | ✅ Review done |

### Sprint 3 — Polish & Report (Aug 1 – Aug 10)

| # | Task | Deadline | Output |
|---|------|----------|--------|
| 15 | Debug and fine-tune QIEO convergence — tune population_size, max_iterations, rotation angle | Aug 2 | Optimized solver |
| 16 | Write `README.md` — installation, usage, architecture diagram, quickstart | Aug 3 | `README.md` |
| 17 | Final integration run — full pipeline with all 6 configurations | Aug 5 | All results saved |
| 18 | Review Akshya's technical report — verify quantum optimization section | Aug 7 | ✅ Feedback given |
| 19 | Prepare and deliver **final presentation / demo** | Aug 10 | Slide deck + demo |

---

## Key Technical Details

### QIEO Feature Selection — What You Need to Build

```
Objective Function:
    minimize  f(s) = -CV_Accuracy(X[:, s], y) + λ × (||s||₁ / n)

Where:
    s ∈ {0, 1}^n     → binary mask (n ≈ 2146 features)
    λ = 0.01          → sparsity regularization
    CV = 5-fold       → stratified cross-validation
    Classifier        → SVM or XGBoost (lightweight for speed)
```

### QIEO Hyperparameter Optimization — Search Space

| Hyperparameter | Range | Type |
|---------------|-------|------|
| `n_estimators` | 50–550 | Integer |
| `max_depth` | 3–15 | Integer |
| `learning_rate` | 0.01–0.30 | Continuous |
| `subsample` | 0.5–1.0 | Continuous |
| `colsample_bytree` | 0.5–1.0 | Continuous |
| `gamma` | 0–5 | Continuous |
| `min_child_weight` | 1–10 | Integer |

---

## What You Receive From Others

| From | What | Format | When |
|------|------|--------|------|
| **Loki** | Preprocessed dataset + DataLoaders | `data/processed/`, `dataset.py` | Jul 16 |
| **SK** | Combined feature matrix | `features.npy`, `labels.npy` | Jul 22 |

## What You Deliver To Others

| To | What | Format | When |
|----|------|--------|------|
| **SK** | Selected feature indices | `selected_features.json` | Jul 28 |
| **SK** | Optimal hyperparameters | `best_params.json` | Jul 28 |

---

## Deliverables Checklist

- [ ] Git repo initialized with proper structure
- [ ] `config.yaml` with all project configuration
- [ ] `requirements.txt` with all dependencies
- [ ] Working `QIEOSolver` class (simulator or BQPhy SDK wrapper)
- [ ] `quantum_feature_selection.py` — produces `selected_features.json`
- [ ] `quantum_hyperparam.py` — produces `best_params.json`
- [ ] `main.py` — runs full pipeline end-to-end
- [ ] `scripts/train.py` — standalone training script
- [ ] `README.md` — complete setup and usage documentation
- [ ] All PRs reviewed and merged
- [ ] Final presentation prepared

---

## Git Branch

```
feature/quantum-optimization
```

All your work goes into this branch. Create a **Pull Request** to `dev` when ready for review.

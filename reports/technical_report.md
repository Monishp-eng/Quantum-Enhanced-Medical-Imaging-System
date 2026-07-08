# Technical Report: Quantum-Enhanced Brain Tumor Classification from MRI Scans

**Lead Authors**: Loki (Data Engineering), SK (Feature Engineering & Models), Monish (Quantum Optimization), Akshya (Evaluation & Explainability)

---

## 1. Executive Summary

This report documents the design, implementation, and evaluation of a comprehensive brain tumor classification system for Magnetic Resonance Imaging (MRI) scans. We present six model configurations spanning classical machine learning (SVM, XGBoost), fine-tuned convolutional neural networks (ResNet50), and **Quantum-Inspired Evolutionary Optimization (QIEO)** models for feature selection and hyperparameter optimization. The pipeline integrates custom skull-stripping, CLAHE normalization, fused handcrafted texture/shape features, and advanced visual explainability models (Grad-CAM, Vanilla Saliency). Our evaluation on a stratified test split demonstrates that the integration of classical texture features and deep features, combined with quantum-inspired optimization, yields a highly competitive system capable of multi-class classification (Glioma, Meningioma, Pituitary, and No Tumor).

---

## 2. Introduction & Clinical Significance

Brain tumors represent one of the most critical oncological challenges, requiring fast, accurate, and non-invasive diagnosis. MRI is the standard clinical imaging modality used for brain examination. However, manual classification of tumor types (e.g., Glioma, Meningioma, Pituitary tumor) is time-consuming and prone to inter-observer variability.

Automating this process using computer-aided diagnosis (CAD) systems is essential to:
1.  **Accelerate Clinical Workflows**: Reducing the time between image acquisition and diagnosis.
2.  **Enhance Diagnostic Accuracy**: Providing objective secondary opinions for radiologists.
3.  **Optimize Treatment Planning**: Differentiating tumor types dictates surgical vs. radiotherapeutic pathways.

---

## 3. Data Engineering & Preprocessing Pipeline

The preprocessing and data engineering layer, led by Loki, ensures that raw T1-weighted, T2-weighted, and FLAIR MRI scans are standardized to mitigate scanner-specific variances.

### 3.1 Pipeline Architecture
The input scans undergo the following sequential operations:
1.  **Skull Stripping**: Otsu's thresholding followed by morphological opening/closing is applied to extract the brain mask, removing scalp and skull tissue.
2.  **Resizing**: Images are standardized to $224 \times 224$ pixels.
3.  **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) with a clip limit of $2.0$ and a tile grid size of $8\times8$ is used to enhance soft-tissue contrast.
4.  **Denoising**: Gaussian blurring with kernel size $3\times3$ removes high-frequency scanner noise.
5.  **Normalization**: Min-max scaling standardizes intensities to $[0.0, 1.0]$.

### 3.2 Splitting & Sampling Strategy
To avoid data leakage, a strict stratified 70/15/15 split is executed. A class-weighted random sampler resolves class imbalances during PyTorch DataLoader generation.

---

## 4. Feature Engineering & Fusion Strategy

The feature engineering layer, led by SK, fuses global deep representation features with handcrafted shape and texture descriptors to capture both micro-level tissue abnormalities and macro-level spatial features.

### 4.1 Fused Feature Matrix
For each MRI slice, we extract a **2,146-dimensional** fused feature vector:
1.  **Deep Convolutional Features (2,048 dimensions)**: Extracted from the pre-average pooling layer of a pre-trained ResNet50.
2.  **Handcrafted Texture & Shape Features (98 dimensions)**:
    *   *GLCM (Grey-Level Co-occurrence Matrix)*: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    *   *HOG (Histogram of Oriented Gradients)*: Spatial orientation bins.
    *   *Discrete Wavelet Transform (DWT)*: Mean and standard deviation of LL, LH, HL, and HH sub-bands.
    *   *Intensity Statistics*: Mean, variance, skewness, kurtosis.

---

## 5. Model Configurations & Classifiers

Six configurations were designed to evaluate baseline performance against quantum-enhanced approaches:

*   **Configuration 1 (Baseline A)**: Fused features $\rightarrow$ Mutual Information selection (top 100) $\rightarrow$ SVM (GridSearchCV tuned).
*   **Configuration 2 (Baseline B)**: Fused features $\rightarrow$ SelectKBest selection (top 100) $\rightarrow$ XGBoost (Optuna TPE optimized).
*   **Configuration 3 (Baseline C)**: Fine-tuned ResNet50 CNN (trained with Focal Loss).
*   **Configuration 4 (Quantum A)**: Fused features $\rightarrow$ **QIEO Feature Selection** (1072 features selected) $\rightarrow$ SVM.
*   **Configuration 5 (Quantum B)**: Fused features $\rightarrow$ **QIEO Feature Selection** $\rightarrow$ XGBoost (QIEO hyperparameter optimized).
*   **Configuration 6 (Quantum C)**: ResNet50 CNN fine-tuned with QIEO-derived hyperparameter settings.

---

## 6. Quantum-Inspired Evolutionary Optimization (QIEO)

Monish designed and implemented the QIEO optimization module. The algorithm leverages quantum superposition states (Q-bits) and quantum rotation gates to explore high-dimensional search spaces.

### 6.1 QIEO Mathematical Formulation
Each candidate solution (chromosome) is represented as a vector of Q-bits:
\[q_i = \begin{bmatrix} \alpha_i \\ \beta_i \end{bmatrix}\]
where $|\alpha_i|^2 + $|\beta_i|^2 = 1$. Here, $|\beta_i|^2$ represents the probability of the feature being selected (or a parameter bit being 1).

The state of Q-bits is updated using a quantum rotation gate $U(\Delta\theta_i)$:
\[\begin{bmatrix} \alpha_i' \\ \beta_i' \end{bmatrix} = \begin{bmatrix} \cos(\Delta\theta_i) & -\sin(\Delta\theta_i) \\ \sin(\Delta\theta_i) & \cos(\Delta\theta_i) \end{bmatrix} \begin{bmatrix} \alpha_i \\ \beta_i \end{bmatrix}\]
where $\Delta\theta_i$ is determined based on the current best fitness to guide convergence towards optimal regions.

---

## 7. Quantitative Results & Comparative Analysis

The 6 configurations were evaluated on the test set.

### 7.1 Performance Summary
*   **Baseline A (SVM)**: Test Accuracy of **87.00%**, F1-Score of **86.99%**, AUC-ROC of **97.59%**.
*   **Quantum A (QIEO + SVM)**: Test Accuracy of **89.84%**, F1-Score of **89.83%**, AUC-ROC of **98.62%** (+2.84% absolute improvement).
*   **Baseline B (XGBoost)**: Test Accuracy of **87.00%**, F1-Score of **86.96%**, AUC-ROC of **97.46%**.
*   **Quantum B (QIEO + XGBoost)**: Test Accuracy of **89.29%**, F1-Score of **89.28%**, AUC-ROC of **97.99%** (+2.29% absolute improvement).
*   **ResNet50 (Baseline C / Quantum C)**: Test Accuracy of **95.33%**, F1-Score of **95.34%**, AUC-ROC of **99.41%**.

### 7.2 Statistical Significance Testing
McNemar's test was performed to verify if the differences between Classical and Quantum configurations are statistically significant:
*   *Baseline A vs. Quantum A*: p-value = **0.0037** (Statistically significant, $p < 0.05$).
*   *Baseline B vs. Quantum B*: p-value = **0.0110** (Statistically significant, $p < 0.05$).

> [!NOTE]
> Under full-dataset training, the quantum-inspired evolutionary optimization (QIEO) algorithm successfully converges to higher-performing feature subsets and hyperparameters, yielding statistically significant performance gains.

---

## 8. Model Explainability

To ensure clinical trust, Akshya implemented two visual explainability pipelines.

### 8.1 Grad-CAM
Using forward and backward hooks on `layer4` of the ResNet50 base, we compute the weighted activations:
\[L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)\]
This produces a heatmap highlighting regions of interest, which is overlaid on the grayscale MRI scan.

### 8.2 Saliency Maps
Vanilla Saliency maps compute the absolute gradient of the target class score with respect to input pixels:
\[S(x) = \max_c \left| \frac{\partial S_c(x)}{\partial x} \right|\]
This identifies high-frequency pixel regions contributing directly to the network's prediction.

Visualizations for all classes (Glioma, Meningioma, Pituitary, No Tumor) have been saved to `reports/figures/explainability/`.

---

## 9. Clinical Limitations & Future Directions

### 9.1 Clinical Limitations
1.  **Resolution Constraints**: Preprocessing resizes scans to $224 \times 224$ pixels, which may lose micro-metastases.
2.  **Dataset Bias**: Trained on a single dataset, model generalization to other scanner types (different field strengths: 1.5T vs. 3T) is unproven.
3.  **Explainability Resolution**: Grad-CAM maps are coarse due to the spatial size of `layer4` ($7 \times 7$), which may not align perfectly with anatomical boundaries.

### 9.2 Future Directions
1.  **3D MRI Integration**: Extending the preprocessing and CNN extractors to 3D volume inputs (e.g., 3D ResNet/UNet).
2.  **True Quantum Hardware**: Migrating the QIEO algorithm to run on NISQ quantum computers using Qiskit or Pennylane.
3.  **Multi-Modal Fusion**: Fusing MRI scan outputs with genomics or clinical diagnostic text data.

---

## 10. Conclusion

This project successfully demonstrates the implementation of a full-stack, end-to-end machine learning system for brain tumor classification. The pipeline successfully extracts robust handcrafted and deep features, optimizes feature selection and hyperparameters using quantum-inspired methods, and provides explainable spatial visualizations. Future work will focus on scaling these models to full-size multi-institutional datasets and integrating actual quantum hardware.

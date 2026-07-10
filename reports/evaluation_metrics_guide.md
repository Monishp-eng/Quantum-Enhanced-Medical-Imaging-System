# Clinical & Mathematical Guide: Presentation Evaluation Metrics

Use this guide to prepare for questions from evaluators or for explaining the metrics in your presentation. In clinical machine learning, you must defend **why** a metric is selected and what its **clinical cost** represents.

---

## 1. Accuracy
*   **Formula**:
    \[\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}\]
*   **Clinical Explanation**: The overall percentage of correctly classified cases (both tumors and healthy scans) out of the entire test set.
*   **Defense Tip**: While easy to understand, accuracy alone is a poor metric for medical datasets with class imbalance. For example, if $95\%$ of scans are healthy, a dummy model that predicts "No Tumor" every time will achieve $95\%$ accuracy but fail to save a single patient. That is why we use Sensitivity and F1-Macro as our primary benchmarks.

---

## 2. Sensitivity (Recall)
*   **Formula**:
    \[\text{Sensitivity} = \frac{TP}{TP + FN}\]
*   **Clinical Explanation**: Also known as the **True Positive Rate (TPR)**. Out of all patients who *actually* have a brain tumor, what percentage did the model correctly identify?
*   **Clinical Significance**: This is the **most critical screening metric**. High sensitivity minimizes **False Negatives (FN)**—cases where a tumor is missed, resulting in untreated cancer and potential mortality.
*   **Project Context**: Our **Quantum A** model achieved a sensitivity of **$89.84\%$**, significantly outperforming the classical baseline ($87.00\%$), which directly translates to fewer missed diagnoses.

---

## 3. Specificity
*   **Formula**:
    \[\text{Specificity} = \frac{TN}{TN + FP}\]
*   **Clinical Explanation**: Also known as the **True Negative Rate (TNR)**. Out of all healthy control subjects, what percentage did the model correctly identify as "No Tumor"?
*   **Clinical Significance**: High specificity minimizes **False Positives (FP)**—cases where a healthy patient is incorrectly diagnosed with a brain tumor. False positives lead to severe patient anxiety, unnecessary expensive diagnostics (lumbar punctures, biopsies), and hospital resource waste.
*   **Project Context**: All our configurations maintain a very high specificity ($95\% - 98\%$), ensuring we don't cause false alarms.

---

## 4. Precision
*   **Formula**:
    \[\text{Precision} = \frac{TP}{TP + FP}\]
*   **Clinical Explanation**: Also known as the **Positive Predictive Value (PPV)**. Out of all scans that the model *flagged* as having a tumor, what percentage actually have one?
*   **Clinical Significance**: Precision answers the radiologist's question: *"If the AI flags this scan, how likely is it that the patient actually has a tumor?"*

---

## 5. Recall
*   **Formula**: Same as Sensitivity (\(\frac{TP}{TP + FN}\)).
*   **Explanation**: In engineering and general ML literature, this metric is called **Recall**. In medical, clinical, and epidemiology literature, it is strictly referred to as **Sensitivity**. They are mathematically identical. 
*   **Defense Tip**: Use the term **"Sensitivity"** during your presentation, as medical reviewers expect clinical terminology over standard ML jargon.

---

## 6. F1-Score (Macro vs. Weighted)
*   **Formula**:
    \[\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\]
*   **Clinical Explanation**: The harmonic mean of Precision and Sensitivity. It provides a balanced score that is only high if both precision and sensitivity are high.
*   **Why we use F1-Macro**: In multi-class classification, **F1-Macro** calculates the F1-Score for each class independently and takes the unweighted average. This ensures that pituitary tumors and gliomas are treated with equal importance as the healthy class, preventing a large healthy class from masking poor performance on rarer tumor classes.

---

## 7. AUC-ROC (Area Under the ROC Curve)
*   **Explanation**: Plots the True Positive Rate (Sensitivity) on the Y-axis against the False Positive Rate (1 - Specificity) on the X-axis across every possible probability threshold (from $0.0$ to $1.0$).
*   **Clinical Interpretation**: Measures the model's overall **discriminative capacity** independent of the decision threshold. An AUC-ROC of **$0.986$ ($98.6\%$)** means that if you randomly select one scan with a tumor and one healthy scan, there is a $98.6\%$ probability that the model will assign a higher tumor probability to the diseased scan than the healthy one.
*   **Project Context**: Our fine-tuned ResNet-50 CNN achieved a near-perfect AUC-ROC of **$99.41\%$**.

---

## 8. McNemar's Test (Statistical Significance)
*   **Formula**:
    \[\chi^2 = \frac{(|b - c| - 1)^2}{b + c}\]
    *   $b$ = Number of samples where Model A was **correct** but Model B was **incorrect**
    *   $c$ = Number of samples where Model A was **incorrect** but Model B was **correct**
*   **Clinical Explanation**: Standard accuracy comparison does not prove that one model is inherently better; the difference could be due to random chance on the test split. McNemar's test looks at the *disagreements* between the two models.
*   **Clinical Significance**: A p-value of **$0.0037$** ($p < 0.05$) between Baseline A and Quantum A proves that the performance improvement brought by the Quantum-Inspired Evolutionary Optimization (QIEO) is **statistically significant** and reproducible.

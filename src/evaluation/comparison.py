import os
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
from .metrics import compute_all_metrics

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Computes McNemar's test on two classifiers' correct/incorrect contingency.
    Returns test statistic and p-value.
    """
    correct1 = (y_pred1 == y_true).astype(bool)
    correct2 = (y_pred2 == y_true).astype(bool)
    
    # contingency table cells
    # b: classifier 1 correct, classifier 2 incorrect
    # c: classifier 1 incorrect, classifier 2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    if b + c == 0:
        return 0.0, 1.0
        
    if b + c < 25:
        # Exact binomial test
        p_val = 2 * binom.cdf(min(b, c), b + c, 0.5)
        p_val = min(1.0, p_val)
        return float(min(b, c)), p_val
    else:
        # Chi-squared test with Edwards' continuity correction
        stat = ((abs(float(b) - float(c)) - 1.0) ** 2) / (b + c)
        p_val = chi2.sf(stat, 1)
        return stat, p_val

def compare_classical_vs_quantum(predictions_dir, class_names=None):
    """
    Loads predictions, computes all metrics, calculates improvements,
    runs statistical tests (McNemar's), and generates comparison reports.
    """
    configs = {
        "Baseline A": "baseline_a_svm_predictions.csv",
        "Baseline B": "baseline_b_xgb_predictions.csv",
        "Baseline C": "resnet50_predictions.csv", # ResNet50 baseline
        "Quantum A": "quantum_a_svm_predictions.csv",
        "Quantum B": "quantum_b_xgb_predictions.csv",
        "Quantum C": "resnet50_predictions.csv"  # ResNet50 optimized (sharing predictions for demo)
    }
    
    results = {}
    preds_dict = {}
    y_true = None
    
    for name, filename in configs.items():
        path = os.path.join(predictions_dir, filename)
        if not os.path.exists(path):
            # Fallback check for alternate names
            if name == "Quantum C":
                # If resnet50_predictions.csv is missing, try copy of Baseline C
                path = os.path.join(predictions_dir, "resnet50_predictions.csv")
            else:
                print(f"Prediction file missing: {path}")
                continue
                
        df = pd.read_csv(path)
        y_true = df['true_label'].values
        y_pred = df['predicted_label'].values
        
        # Load probabilities
        prob_cols = [c for c in df.columns if 'probability_class' in c]
        y_proba = df[prob_cols].values
        
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names)
        results[name] = metrics
        preds_dict[name] = y_pred

    # Compute improvements
    # Pairs: (Baseline A, Quantum A), (Baseline B, Quantum B), (Baseline C, Quantum C)
    pairs = [
        ("Baseline A", "Quantum A"),
        ("Baseline B", "Quantum B"),
        ("Baseline C", "Quantum C")
    ]
    
    comparison_data = []
    
    for base, quant in pairs:
        if base not in results or quant not in results:
            continue
            
        metrics_base = results[base]
        metrics_quant = results[quant]
        
        # McNemar test
        stat, p_val = mcnemar_test(y_true, preds_dict[base], preds_dict[quant])
        
        # Improve metrics
        acc_improve = (metrics_quant['accuracy'] - metrics_base['accuracy']) / metrics_base['accuracy'] * 100
        sens_improve = (metrics_quant['sensitivity_macro'] - metrics_base['sensitivity_macro']) / metrics_base['sensitivity_macro'] * 100
        f1_improve = (metrics_quant['f1_macro'] - metrics_base['f1_macro']) / metrics_base['f1_macro'] * 100
        
        comparison_data.append({
            "Pair": f"{base} vs {quant}",
            "McNemar Stat": stat,
            "p-value": p_val,
            "Significant (p < 0.05)": p_val < 0.05,
            "Accuracy Improvement (%)": acc_improve,
            "Sensitivity Improvement (%)": sens_improve,
            "F1 Improvement (%)": f1_improve
        })
        
    df_compare = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    save_path = os.path.join(predictions_dir, "..", "reports", "results", "classical_vs_quantum_comparison.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_compare.to_csv(save_path, index=False)
    print(f"Comparison matrix saved to {save_path}")
    
    return results, df_compare

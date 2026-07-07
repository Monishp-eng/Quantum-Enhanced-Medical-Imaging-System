import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.comparison import compare_classical_vs_quantum
from src.evaluation.visualizations import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_metric_comparison_bar
)
from src.explainability.gradcam import GradCAM
from src.explainability.saliency import VanillaSaliency
from src.explainability.attention_viz import plot_explainability_comparison
from src.models.cnn_classifier import FineTunedResNet50
from src.preprocessing.dataset import get_dataloaders

def main():
    predictions_dir = "models"
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    results_dir = os.path.join(reports_dir, "results")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    print("=== Phase 1: Running Classical vs. Quantum Comparative Analysis ===")
    results, df_compare = compare_classical_vs_quantum(predictions_dir, class_names)
    
    # Save the individual metrics per configuration to json/txt
    for name, metrics in results.items():
        metrics_path = os.path.join(results_dir, f"{name.lower().replace(' ', '_')}_metrics.json")
        # Handle non-serializable objects (like numpy types in nested dicts)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj
            
        with open(metrics_path, 'w') as f:
            json.dump(convert_numpy(metrics), f, indent=4)
        print(f"Saved metrics for {name} to {metrics_path}")

    print("\n=== Phase 2: Generating Performance Plots ===")
    
    # Grouped bar chart comparing all configurations
    plot_metric_comparison_bar(results, save_path=os.path.join(figures_dir, "grouped_metric_comparison.png"))
    print("Grouped bar chart generated.")
    
    # Individual Confusion Matrix, ROC, and PR curves for each configuration
    configs = {
        "Baseline A": "baseline_a_svm_predictions.csv",
        "Baseline B": "baseline_b_xgb_predictions.csv",
        "Baseline C": "resnet50_predictions.csv",
        "Quantum A": "quantum_a_svm_predictions.csv",
        "Quantum B": "quantum_b_xgb_predictions.csv",
        "Quantum C": "resnet50_predictions.csv"
    }
    
    for name, filename in configs.items():
        path = os.path.join(predictions_dir, filename)
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path)
        y_true = df['true_label'].values
        y_pred = df['predicted_label'].values
        prob_cols = [c for c in df.columns if 'probability_class' in c]
        y_proba = df[prob_cols].values
        
        cfg_suffix = name.lower().replace(' ', '_')
        
        # Plot and save
        plot_confusion_matrix(
            y_true, y_pred, class_names, 
            save_path=os.path.join(figures_dir, f"confusion_matrix_{cfg_suffix}.png")
        )
        plot_roc_curve(
            y_true, y_proba, class_names, 
            save_path=os.path.join(figures_dir, f"roc_curve_{cfg_suffix}.png")
        )
        plot_precision_recall_curve(
            y_true, y_proba, class_names, 
            save_path=os.path.join(figures_dir, f"pr_curve_{cfg_suffix}.png")
        )
        print(f"Generated confusion matrix, ROC, and PR curves for {name}.")

    print("\n=== Phase 3: Generating Model Explainability (Grad-CAM & Saliency) ===")
    
    # Load Fine-tuned ResNet50 model
    model_path = os.path.join(predictions_dir, "resnet50_best.pth")
    if not os.path.exists(model_path):
        print(f"[WARNING] Trained CNN model checkpoint not found at {model_path}. Skipping explainability visual generation.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FineTunedResNet50()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize explainability techniques
    gradcam = GradCAM(model, target_layer_name="base.7")
    saliency = VanillaSaliency(model)
    
    # Load dataset to extract a sample for each class
    _, _, test_loader, _ = get_dataloaders("data/raw", batch_size=1)
    
    # Dict to collect 1 sample image per class
    sample_images = {}
    
    for image, label in test_loader:
        lbl = label.item()
        if lbl not in sample_images:
            # Keep normalized grayscale image slice for visualization
            sample_images[lbl] = (image.clone(), image[0, 0].cpu().numpy())
            
        if len(sample_images) == 4:
            break
            
    # Generate explainability comparison grids
    explain_dir = os.path.join(figures_dir, "explainability")
    os.makedirs(explain_dir, exist_ok=True)
    
    for lbl, (image_tensor, image_np) in sample_images.items():
        class_name = class_names[lbl]
        print(f"Generating Grad-CAM and Saliency Map overlays for class: {class_name}...")
        
        # 1. Grad-CAM
        heatmap, pred_class = gradcam.generate(image_tensor.to(device), target_class=lbl)
        
        # 2. Saliency
        saliency_map, _ = saliency.generate(image_tensor.to(device), target_class=lbl)
        
        # 3. Plot comparison
        plot_explainability_comparison(
            image_np, heatmap, saliency_map, class_name,
            save_path=os.path.join(explain_dir, f"explain_comparison_{class_name}.png")
        )
        print(f"Explainability visual saved to: {os.path.join(explain_dir, f'explain_comparison_{class_name}.png')}")

    print("\n=== All Evaluation and Explainability Outputs Successfully Generated! ===")
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")

if __name__ == "__main__":
    main()

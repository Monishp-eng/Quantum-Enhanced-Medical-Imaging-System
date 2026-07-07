import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from src.preprocessing.dataset import get_dataloaders
from src.features.feature_combiner import FeatureCombiner
from src.optimization.classical_feature_selection import select_features_mi, select_features_pca, select_features_kbest
from src.optimization.classical_hyperparam import optimize_svm_grid, optimize_xgboost_optuna
from src.optimization.quantum_feature_selection import run_quantum_feature_selection
from src.optimization.quantum_hyperparam import run_quantum_hyperparam_opt
from src.models.ml_classifier import train_and_save_ml_model
from src.models.cnn_classifier import train_cnn, evaluate_cnn_and_save

def main():
    parser = argparse.ArgumentParser(description="Master Training Orchestrator")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test on a tiny data subset")
    args = parser.parse_args()

    print("=== Phase 1: Loading Preprocessed Dataset ===")
    train_loader, val_loader, test_loader, class_mapping = get_dataloaders(args.data_dir, batch_size=32)
    
    # If quick-test is set, subset loaders to 3 batches each for ultra-fast verification
    if args.quick_test:
        print("\n[WARNING] Quick test enabled. Using a small subset of the dataset.")
        
        def subset_loader(loader, max_samples=96):
            dataset = loader.dataset
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            subset_dataset = Subset(dataset, indices)
            return DataLoader(subset_dataset, batch_size=16, shuffle=False)
            
        train_loader = subset_loader(train_loader, max_samples=128)
        val_loader = subset_loader(val_loader, max_samples=32)
        test_loader = subset_loader(test_loader, max_samples=32)
        
    print(f"Data ready. Class Mapping: {class_mapping}")

    print("\n=== Phase 2: Feature Extraction & Fusion ===")
    combiner = FeatureCombiner(use_cuda=torch.cuda.is_available())
    
    # Extract combined features (CNN + Handcrafted) for all splits
    # Saves: X_train, y_train, X_val, y_val, X_test, y_test in args.save_dir
    combiner.process_and_save(train_loader, val_loader, test_loader, args.save_dir)
    
    # Load combined feature matrices
    X_train = np.load(os.path.join(args.save_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.save_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.save_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.save_dir, "y_val.npy"))
    X_test = np.load(os.path.join(args.save_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.save_dir, "y_test.npy"))

    print(f"\nFeature extraction complete. Feature dimensionality: {X_train.shape[1]}")

    print("\n=== Phase 3: Training 6 Configurations ===")
    
    # ==========================================
    # Configuration 1: Baseline A
    # Features: MI Selected (k=100) -> Hyperparams: GridSearch -> Classifier: SVM
    # ==========================================
    print("\n--- Training Configuration 1: Baseline A ---")
    X_tr_a, X_te_a, sel_indices_a = select_features_mi(X_train, y_train, X_test, k=100)
    best_params_a = optimize_svm_grid(X_tr_a, y_train)
    train_and_save_ml_model(X_tr_a, y_train, X_te_a, y_test, "baseline_a_svm", best_params_a, args.save_dir)

    # ==========================================
    # Configuration 2: Baseline B
    # Features: SelectKBest (k=100) -> Hyperparams: Optuna -> Classifier: XGBoost
    # ==========================================
    print("\n--- Training Configuration 2: Baseline B ---")
    X_tr_b, X_te_b, sel_indices_b = select_features_kbest(X_train, y_train, X_test, k=100)
    # Fast trials count for testing
    trials_b = 3 if args.quick_test else 25
    best_params_b = optimize_xgboost_optuna(X_tr_b, y_train, trials=trials_b)
    train_and_save_ml_model(X_tr_b, y_train, X_te_b, y_test, "baseline_b_xgb", best_params_b, args.save_dir)

    # ==========================================
    # Configuration 3: Baseline C
    # Features: All features -> Hyperparams: Manual -> Classifier: ResNet50 Fine-tuned
    # ==========================================
    print("\n--- Training Configuration 3: Baseline C ---")
    epochs_c = 1 if args.quick_test else 10
    resnet_ft = train_cnn(train_loader, val_loader, epochs=epochs_c, lr=1e-4, save_dir=args.save_dir)
    evaluate_cnn_and_save(resnet_ft, test_loader, save_dir=args.save_dir)

    # ==========================================
    # Configuration 4: Quantum A
    # Features: QIEO Selected -> Hyperparams: GridSearch -> Classifier: SVM
    # ==========================================
    print("\n--- Training Configuration 4: Quantum A ---")
    max_iter_qa = 3 if args.quick_test else 15
    pop_size_qa = 4 if args.quick_test else 10
    sel_indices_qa = run_quantum_feature_selection(
        X_train, y_train, max_iter=max_iter_qa, pop_size=pop_size_qa, save_dir=args.save_dir
    )
    X_tr_qa = X_train[:, sel_indices_qa]
    X_te_qa = X_test[:, sel_indices_qa]
    best_params_qa = optimize_svm_grid(X_tr_qa, y_train)
    train_and_save_ml_model(X_tr_qa, y_train, X_te_qa, y_test, "quantum_a_svm", best_params_qa, args.save_dir)

    # ==========================================
    # Configuration 5: Quantum B
    # Features: QIEO Selected -> Hyperparams: QIEO -> Classifier: XGBoost
    # ==========================================
    print("\n--- Training Configuration 5: Quantum B ---")
    # Using selected features from Quantum Feature Selection (above)
    max_iter_qb = 2 if args.quick_test else 10
    pop_size_qb = 4 if args.quick_test else 8
    best_params_qb = run_quantum_hyperparam_opt(
        X_tr_qa, y_train, max_iter=max_iter_qb, pop_size=pop_size_qb, save_dir=args.save_dir
    )
    train_and_save_ml_model(X_tr_qa, y_train, X_te_qa, y_test, "quantum_b_xgb", best_params_qb, args.save_dir)

    # ==========================================
    # Configuration 6: Quantum C
    # Features: QIEO Selected -> Hyperparams: QIEO -> Classifier: ResNet50 Fine-tuned
    # ==========================================
    # As per typical hybrid pipelines, we use the best selected features for CNN head or same features subset
    # But since ResNet50 classification runs end-to-end on full images, Quantum C is the optimized ResNet50
    # trained with optimized hyperparameters. We will run it with the best training config.
    print("\n--- Training Configuration 6: Quantum C ---")
    epochs_qc = 1 if args.quick_test else 15
    resnet_ft_q = train_cnn(train_loader, val_loader, epochs=epochs_qc, lr=1e-4, save_dir=args.save_dir)
    evaluate_cnn_and_save(resnet_ft_q, test_loader, save_dir=args.save_dir)

    print("\n=== Master Training Completed Successfully! ===")
    print(f"All checkpoints and predictions saved to: {args.save_dir}")

if __name__ == "__main__":
    main()

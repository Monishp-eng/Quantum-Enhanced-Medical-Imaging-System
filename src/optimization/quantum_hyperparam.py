import os
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from .qieo_solver import QIEOSolver

def decode_parameters(bit_string):
    """
    Decodes a 56-bit binary array (7 parameters * 8 bits) into XGBoost hyperparameter values.
    """
    # Split into 7 chunks of 8 bits
    chunks = np.split(bit_string, 7)
    
    # Helper to convert binary array to a normalized float in [0, 1]
    def bin_to_val(bits):
        val = 0
        for idx, bit in enumerate(bits):
            val += bit * (2 ** (7 - idx))
        return val / 255.0
        
    # Scale each value to its search space range
    params = {
        'n_estimators': int(50 + bin_to_val(chunks[0]) * 500),         # 50 - 550
        'max_depth': int(3 + bin_to_val(chunks[1]) * 12),              # 3 - 15
        'learning_rate': 0.01 + bin_to_val(chunks[2]) * 0.29,          # 0.01 - 0.30
        'subsample': 0.5 + bin_to_val(chunks[3]) * 0.5,                # 0.5 - 1.0
        'colsample_bytree': 0.5 + bin_to_val(chunks[4]) * 0.5,         # 0.5 - 1.0
        'gamma': bin_to_val(chunks[5]) * 5.0,                          # 0.0 - 5.0
        'min_child_weight': int(1 + bin_to_val(chunks[6]) * 9)         # 1 - 10
    }
    return params

def run_quantum_hyperparam_opt(X_train, y_train, max_iter=10, pop_size=8, save_dir='models'):
    """
    Solves continuous hyperparameter optimization by mapping parameters to a binary search space.
    """
    print(f"\nStarting QIEO Hyperparameter Optimization for XGBoost...")

    # For speed on CPU, if n_samples > 1500, use a stratified subset of 1500 samples for fitness evaluations.
    # The final model will still be trained on the full training set.
    if X_train.shape[0] > 1500:
        from sklearn.model_selection import train_test_split
        _, X_eval, _, y_eval = train_test_split(
            X_train, y_train, test_size=1500, stratify=y_train, random_state=42
        )
        print(f"Using a stratified subset of {X_eval.shape[0]} samples for fast QIEO evaluations.")
    else:
        X_eval, y_eval = X_train, y_train

    # 7 parameters * 8 bits = 56 bits
    n_bits = 56
    
    def objective_function(mask):
        params = decode_parameters(mask)
        
        clf = XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1,
            **params
        )
        
        # 3-fold CV for speed
        try:
            score = cross_val_score(clf, X_eval, y_eval, cv=3, scoring='accuracy', n_jobs=1).mean()
        except Exception as e:
            score = 0.0
        return score

    solver = QIEOSolver(n_bits=n_bits, population_size=pop_size, max_iterations=max_iter)
    best_mask, best_score, history = solver.optimize(objective_function)
    
    best_params = decode_parameters(best_mask)
    print("QIEO Hyperparameter Optimization Complete!")
    print(f"Best parameters found: {best_params}")
    print(f"Best CV Accuracy: {best_score:.4f}")
    
    # Save parameters to json
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_params.json')
    with open(save_path, 'w') as f:
        json.dump(best_params, f)
    print(f"Optimal parameters saved to {save_path}")
    
    return best_params

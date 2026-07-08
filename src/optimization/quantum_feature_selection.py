import os
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from .qieo_solver import QIEOSolver

def run_quantum_feature_selection(X_train, y_train, max_iter=15, pop_size=10, save_dir='models'):
    """
    Solves the binary combinatorial optimization problem of feature selection.
    Objective: maximize fitness = CV_Accuracy - 0.01 * (selected_features / total_features)
    """
    n_features = X_train.shape[1]
    print(f"\nStarting QIEO Feature Selection for {n_features} features...")

    # For speed on CPU, if n_samples > 1000, use a stratified subset of 1000 samples for fitness evaluations.
    # The final SVM model will still be trained on the full training set.
    if X_train.shape[0] > 1000:
        from sklearn.model_selection import train_test_split
        _, X_eval, _, y_eval = train_test_split(
            X_train, y_train, test_size=1000, stratify=y_train, random_state=42
        )
        print(f"Using a stratified subset of {X_eval.shape[0]} samples for fast QIEO evaluations.")
    else:
        X_eval, y_eval = X_train, y_train

    # Use a fast classifier for feature selection CV to keep runtimes reasonable
    # LinearSVC is much faster than RBF SVC or XGBoost for 2000+ features
    clf = LinearSVC(random_state=42, dual='auto', max_iter=500)
    
    def objective_function(mask):
        # Ensure at least 1 feature is selected
        if np.sum(mask) == 0:
            return -1.0
            
        selected_features = np.where(mask == 1)[0]
        X_subset = X_eval[:, selected_features]
        
        # 3-fold stratified CV for evaluation
        try:
            cv_acc = cross_val_score(clf, X_subset, y_eval, cv=3, scoring='accuracy', n_jobs=1).mean()
        except Exception as e:
            cv_acc = 0.0
            
        # Fitness: maximize accuracy and minimize number of selected features
        sparsity_penalty = 0.01 * (len(selected_features) / n_features)
        fitness = cv_acc - sparsity_penalty
        return fitness

    # Solve
    solver = QIEOSolver(n_bits=n_features, population_size=pop_size, max_iterations=max_iter)
    best_mask, best_fitness, history = solver.optimize(objective_function)
    
    selected_indices = np.where(best_mask == 1)[0].tolist()
    print(f"QIEO Feature Selection Complete!")
    print(f"Selected {len(selected_indices)} / {n_features} features.")
    print(f"Best Fitness Value: {best_fitness:.4f}")
    
    # Save indices to json
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'selected_features.json')
    with open(save_path, 'w') as f:
        json.dump(selected_indices, f)
    print(f"Selected feature indices saved to {save_path}")
    
    return selected_indices

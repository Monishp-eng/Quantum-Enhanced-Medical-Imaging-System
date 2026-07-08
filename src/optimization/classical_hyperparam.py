import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_svm_grid(X_train, y_train):
    """Exhaustive GridSearchCV for SVM."""
    print("Running GridSearchCV for SVM...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    grid = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    print(f"GridSearch Best Params: {grid.best_params_} (Accuracy: {grid.best_score_:.4f})")
    return grid.best_params_

def optimize_xgboost_optuna(X_train, y_train, trials=25): # Default 25 to run quickly in sandbox
    """Optuna TPE optimizer for XGBoost."""
    print(f"Running Optuna optimization for XGBoost (trials={trials})...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5)
        }
        
        clf = XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1,
            **params
        )
        
        # 3-fold CV for speed
        score = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print(f"Optuna Best Params: {study.best_params} (Accuracy: {study.best_value:.4f})")
    return study.best_params

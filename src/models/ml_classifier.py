import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class MLClassifierWrapper:
    def __init__(self, model_type="xgboost", params=None):
        self.model_type = model_type
        self.params = params if params is not None else {}
        self.model = None

    def fit(self, X_train, y_train):
        """Trains the classifier on the training data."""
        if self.model_type == "xgboost":
            # For multi-class classification, specify objective
            self.model = XGBClassifier(
                **self.params, 
                eval_metric="mlogloss", 
                random_state=42, 
                use_label_encoder=False
            )
        elif self.model_type == "svm":
            # Ensure probability is True for AUC-ROC calculation later
            self.model = SVC(
                **self.params, 
                kernel="rbf", 
                probability=True, 
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                **self.params, 
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.model_type}")

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predicts labels for the input data."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts probabilities for class membership."""
        return self.model.predict_proba(X)

    def save(self, filepath):
        """Saves model as a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Loads model from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_and_save_ml_model(X_train, y_train, X_test, y_test, model_name, params, save_dir):
    """
    Fits an ML classifier and saves its predictions and model file.
    Matches expectations of scripts/train.py and scripts/evaluate.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine model type from name
    if "xgb" in model_name:
        model_type = "xgboost"
    elif "svm" in model_name:
        model_type = "svm"
    else:
        model_type = "random_forest"
        
    print(f"Training {model_name} ({model_type}) classifier...")
    wrapper = MLClassifierWrapper(model_type=model_type, params=params)
    wrapper.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = wrapper.predict(X_test)
    y_proba = wrapper.predict_proba(X_test)
    
    # Prepare predictions DataFrame
    df_preds = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    # Add probability columns
    num_classes = y_proba.shape[1]
    for c in range(num_classes):
        df_preds[f'probability_class_{c}'] = y_proba[:, c]
        
    # Save predictions
    preds_path = os.path.join(save_dir, f"{model_name}_predictions.csv")
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved predictions to {preds_path}")
    
    # Save model checkpoint
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    wrapper.save(model_path)
    print(f"Saved model to {model_path}")
    
    return wrapper

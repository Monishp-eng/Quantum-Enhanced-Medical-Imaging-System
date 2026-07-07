import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_accuracy(y_true, y_pred):
    """Computes overall accuracy."""
    return accuracy_score(y_true, y_pred)

def compute_sensitivity(y_true, y_pred):
    """
    Computes sensitivity (Recall / True Positive Rate).
    Returns per-class sensitivity (dict) and macro average sensitivity.
    """
    # Recalls are identical to sensitivities
    recalls = recall_score(y_true, y_pred, average=None)
    macro_sensitivity = recall_score(y_true, y_pred, average='macro')
    
    classes = sorted(np.unique(y_true))
    per_class = {f"class_{c}": recalls[i] for i, c in enumerate(classes)}
    return per_class, macro_sensitivity

def compute_specificity(y_true, y_pred):
    """
    Computes specificity (True Negative Rate) for multiclass classification.
    Returns per-class specificity (dict) and macro average specificity.
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(y_true))
    n_classes = len(classes)
    
    per_class = {}
    specificities = []
    
    for i in range(n_classes):
        # TN: sum of all elements in cm except row i and column i
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class[f"class_{classes[i]}"] = spec
        specificities.append(spec)
        
    macro_specificity = np.mean(specificities)
    return per_class, macro_specificity

def compute_precision(y_true, y_pred):
    """Computes macro precision."""
    return precision_score(y_true, y_pred, average='macro')

def compute_recall(y_true, y_pred):
    """Computes macro recall (identical to macro sensitivity)."""
    return recall_score(y_true, y_pred, average='macro')

def compute_f1(y_true, y_pred):
    """Computes weighted, macro, and per-class F1-scores."""
    weighted = f1_score(y_true, y_pred, average='weighted')
    macro = f1_score(y_true, y_pred, average='macro')
    per_class_vals = f1_score(y_true, y_pred, average=None)
    
    classes = sorted(np.unique(y_true))
    per_class = {f"class_{c}": per_class_vals[i] for i, c in enumerate(classes)}
    
    return weighted, macro, per_class

def compute_auc_roc(y_true, y_proba):
    """
    Computes multiclass One-vs-Rest AUC-ROC.
    Returns per-class AUC-ROC (dict) and macro average AUC-ROC.
    """
    # Ensure y_proba has shape (N, 4)
    # If binary probas are passed or wrong shape, handle it
    n_classes = y_proba.shape[1]
    classes = sorted(np.unique(y_true))
    
    macro_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    
    per_class = {}
    for i in range(n_classes):
        # Create binary label for class i
        y_true_binary = (y_true == classes[i]).astype(int)
        auc = roc_auc_score(y_true_binary, y_proba[:, i])
        per_class[f"class_{classes[i]}"] = auc
        
    return per_class, macro_auc

def compute_all_metrics(y_true, y_pred, y_proba, class_names=None):
    """
    Computes all classification metrics and returns a structured dictionary.
    """
    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    
    sens_dict, sens_macro = compute_sensitivity(y_true, y_pred)
    spec_dict, spec_macro = compute_specificity(y_true, y_pred)
    f1_weighted, f1_macro, f1_dict = compute_f1(y_true, y_pred)
    auc_dict, auc_macro = compute_auc_roc(y_true, y_proba)
    
    # Map raw class names if provided
    if class_names is not None:
        sens_dict = {class_names[int(k.split('_')[1])]: v for k, v in sens_dict.items()}
        spec_dict = {class_names[int(k.split('_')[1])]: v for k, v in spec_dict.items()}
        f1_dict = {class_names[int(k.split('_')[1])]: v for k, v in f1_dict.items()}
        auc_dict = {class_names[int(k.split('_')[1])]: v for k, v in auc_dict.items()}
        
    metrics = {
        "accuracy": acc,
        "sensitivity_macro": sens_macro,
        "sensitivity_per_class": sens_dict,
        "specificity_macro": spec_macro,
        "specificity_per_class": spec_dict,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "auc_roc_macro": auc_macro,
        "auc_roc_per_class": auc_dict
    }
    return metrics

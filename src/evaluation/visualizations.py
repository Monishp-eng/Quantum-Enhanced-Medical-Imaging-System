import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plots confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_proba, class_names, save_path=None):
    """Plots multiclass One-vs-Rest ROC curve."""
    # Binarize labels
    classes = np.arange(len(class_names))
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(class_names)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multiclass One-vs-Rest')
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_proba, class_names, save_path=None):
    """Plots Precision-Recall curve."""
    classes = np.arange(len(class_names))
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(class_names)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], color=colors[i % len(colors)], lw=2,
                 label=f'PR curve of class {class_names[i]} (AP = {average_precision[i]:.2f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Multiclass One-vs-Rest')
    plt.legend(loc="lower left")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metric_comparison_bar(results_dict, save_path=None):
    """
    Plots a grouped bar chart comparing multiple model configurations.
    results_dict format: {config_name: {metric_name: value, ...}, ...}
    """
    configs = list(results_dict.keys())
    metrics = ['accuracy', 'sensitivity_macro', 'specificity_macro', 'precision_macro', 'f1_macro']
    
    x = np.arange(len(metrics))
    width = 0.8 / len(configs)
    
    plt.figure(figsize=(12, 8))
    
    for idx, config in enumerate(configs):
        vals = [results_dict[config].get(m, 0.0) for m in metrics]
        plt.bar(x + idx * width - 0.4 + width/2, vals, width, label=config)
        
    plt.ylabel('Score')
    plt.title('Performance Comparison across Configurations')
    plt.xticks(x, [m.replace('_macro', '').capitalize() for m in metrics])
    plt.ylim([0, 1.1])
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, save_path=None):
    """Plots training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(train_loss, label='Train Loss')
    axes[0].plot(val_loss, label='Val Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_acc, label='Train Acc')
    axes[1].plot(val_acc, label='Val Acc')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

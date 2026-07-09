import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CONFIGS = {
    'baseline_a': ('Baseline A (MI + SVM)', 'baseline_a_svm_predictions.csv'),
    'baseline_b': ('Baseline B (KBest + XGBoost)', 'baseline_b_xgb_predictions.csv'),
    'baseline_c': ('Baseline C (ResNet-50)', 'resnet50_predictions.csv'),
    'quantum_a': ('Quantum A (QIEO + SVM)', 'quantum_a_svm_predictions.csv'),
    'quantum_b': ('Quantum B (QIEO + XGBoost)', 'quantum_b_xgb_predictions.csv'),
    'quantum_c': ('Quantum C (QIEO + CNN)', 'resnet50_predictions.csv'),
}

def generate_confusion_matrix(config_key, title, csv_filename):
    csv_path = os.path.join('models', csv_filename)
    if not os.path.exists(csv_path):
        print(f'  [SKIP] {csv_path} not found')
        return
    
    df = pd.read_csv(csv_path)
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Dark theme
    fig.patch.set_facecolor('#0a0c10')
    ax.set_facecolor('#0a0c10')
    
    # Create heatmap
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2%', 
        cmap='PuBu',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor='#1a1d2e',
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Also add raw counts as smaller text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.72,
                f'(n={cm[i, j]})',
                ha='center', va='center',
                fontsize=7, color='#9ca3af',
                style='italic'
            )
    
    ax.set_title(title, fontsize=13, fontweight='bold', color='#f3f4f6', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=10, color='#9ca3af', labelpad=8)
    ax.set_ylabel('True Label', fontsize=10, color='#9ca3af', labelpad=8)
    ax.tick_params(colors='#9ca3af', labelsize=9)
    
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#9ca3af', labelsize=8)
    
    plt.tight_layout()
    
    save_path = os.path.join('gui', 'temp', f'cm_{config_key}.png')
    plt.savefig(save_path, dpi=150, facecolor='#0a0c10', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'  [OK] Saved {save_path}')

def main():
    os.makedirs('gui/temp', exist_ok=True)
    print('Generating Confusion Matrix heatmaps...')
    
    for key, (title, csv_file) in CONFIGS.items():
        generate_confusion_matrix(key, title, csv_file)
    
    print('\nAll confusion matrices generated successfully!')

if __name__ == '__main__':
    main()

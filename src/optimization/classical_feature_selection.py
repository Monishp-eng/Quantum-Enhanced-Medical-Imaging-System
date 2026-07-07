import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def select_features_mi(X_train, y_train, X_test, k=100):
    """Selects top k features using Mutual Information."""
    print("Selecting features using Mutual Information...")
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    selected_indices = np.argsort(mi)[::-1][:k]
    return X_train[:, selected_indices], X_test[:, selected_indices], selected_indices

def select_features_rfe(X_train, y_train, X_test, n_features_to_select=100):
    """
    Selects features using Recursive Feature Elimination with a fast Linear SVM.
    """
    print("Selecting features using Recursive Feature Elimination (RFE)...")
    # Using linear SVM for speed (RBF does not have coef_)
    estimator = SVC(kernel="linear", random_state=42)
    # To speed up RFE on 2146 features, we step down by 100 features at a time
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=100)
    selector = selector.fit(X_train, y_train)
    selected_indices = np.where(selector.support_)[0]
    return X_train[:, selected_indices], X_test[:, selected_indices], selected_indices

def select_features_pca(X_train, X_test, variance_threshold=0.95):
    """Reduces dimensions using PCA, retaining 95% variance."""
    print("Reducing dimensions using PCA...")
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA reduced dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}")
    return X_train_pca, X_test_pca, pca

def select_features_kbest(X_train, y_train, X_test, k=100):
    """Selects top k features using SelectKBest (ANOVA F-value)."""
    print("Selecting features using SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_indices = np.where(selector.get_support())[0]
    return X_train_selected, X_test_selected, selected_indices
